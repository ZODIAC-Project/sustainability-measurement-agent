"""Execute local treatments with cancellable process groups and durable logs."""
import logging
import math
import os
import signal
import subprocess
import shlex
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from sma.model import SMAObserver, Triggerable

log = logging.getLogger(__name__)


class SubprocessrunnerSmaModule(SMAObserver, Triggerable):
    """Run local commands without a shell and save outputs even when they fail."""

    def __init__(self, config: dict):
        self.config = SubprocessRunnerConfig.from_dict(config)
        self.config.validate()
        # The agent waits this long for cooperative cancellation to finish.
        self.cancellation_grace_period = self.config.termination_grace_seconds + 2.0
        self._logs = tempfile.TemporaryDirectory(prefix="sma-subprocess-")
        self._phase_results = {}
        self._cancelled = False
        self._report = None
        self._output_before = {}
        self._active_cancel = None

    def trigger(self, cancel: threading.Event, **kwargs) -> Optional[Dict[str, Any]]:
        self.cancel = cancel
        self._phase_results.pop("treatment", None)
        for stream in ("stdout", "stderr"):
            (Path(self._logs.name) / f"treatment.{stream}.log").unlink(missing_ok=True)
        self._output_before = {str(path): self._signature(path) for path in self.config.output_paths()}
        if cancel.is_set():
            self._cancelled = True
            return {"subprocess_cancelled": True}
        command = shlex.join(shlex.split(self.config.trigger_command) + self.config.parameter_args())
        exit_code = self._run_subprocess(command, cancel=cancel)
        return {"subprocess_exit_code": exit_code, "subprocess_cancelled": self._cancelled}

    def request_cancel(self):
        """Forward external SMA cancellation to the currently running phase."""
        if self._active_cancel is not None:
            self._active_cancel.set()

    def onSetup(self):
        self._logs.cleanup()
        self._logs = tempfile.TemporaryDirectory(prefix="sma-subprocess-")
        self._phase_results = {}
        self._report = None
        self._output_before = {str(path): self._signature(path) for path in self.config.output_paths()}
        if self.config.setup_command:
            code = self._run_subprocess(self.config.setup_command, phase="setup")
            if code != 0:
                raise RuntimeError(f"Subprocess setup failed with exit code {code}")

    def onTeardown(self):
        try:
            if self.config.teardown_command:
                code = self._run_subprocess(self.config.teardown_command, phase="teardown")
                if code != 0:
                    raise RuntimeError(f"Subprocess teardown failed with exit code {code}")
        finally:
            # Teardown happens after onReport; retain its logs and exit code too.
            if self._report is not None:
                self.onReport(self._report)

    @staticmethod
    def _signature(path):
        if not path.is_file():
            return None
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def onReport(self, report=None):
        if report is None:
            return
        self._report = report
        user_data = report.run_data.user_data
        if user_data is None:
            user_data = report.run_data.user_data = {}
        for phase, code in self._phase_results.items():
            key = "subprocess_exit_code" if phase == "treatment" else f"subprocess_{phase}_exit_code"
            user_data[key] = code
        log_destination = Path(report.location) / "subprocess_logs"
        log_destination.mkdir(parents=True, exist_ok=True)
        for source in Path(self._logs.name).iterdir():
            shutil.copy2(source, log_destination / source.name)
        destination = Path(report.location) / "subprocess_output"
        for source in self.config.output_paths():
            signature = self._signature(source)
            if signature is None or signature == self._output_before.get(str(source)):
                log.warning("No new output file for this run: %s", source)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)

    @staticmethod
    def _signal_process_group(process, sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def _terminate(self, process):
        """Terminate descendants too, including when their parent exits on TERM."""
        self._signal_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + self.config.termination_grace_seconds
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader before checking whether the group remains.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
            elif process.returncode is not None:
                break
            time.sleep(0.02)
        if os.name == "posix":
            self._signal_process_group(process, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        process.wait()

    def _run_subprocess(self, command: str, cancel: Optional[threading.Event] = None,
                        phase: str = "treatment") -> int:
        cancel = cancel if cancel is not None else threading.Event()
        self._cancelled = False
        self._active_cancel = cancel
        log.info("Running subprocess %s", phase)
        # Files avoid pipe deadlocks and unbounded in-memory capture. They also
        # preserve partial binary output when the child must be killed.
        stdout_path = Path(self._logs.name) / f"{phase}.stdout.log"
        stderr_path = Path(self._logs.name) / f"{phase}.stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                shlex.split(command), cwd=self.config.workdir, env=self.config.env,
                stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"),
            )
            try:
                while process.poll() is None:
                    if cancel.wait(0.05):
                        self._cancelled = True
                        self._terminate(process)
                        break
                code = process.wait()
            except BaseException:
                self._terminate(process)
                raise
        self._active_cancel = None
        self._phase_results[phase] = code
        log.info("Subprocess %s finished with exit code %s", phase, code)
        return code


@dataclass
class SubprocessRunnerConfig:
    trigger_command: str
    setup_command: str
    teardown_command: str
    workdir: Path
    treatment_parameters: Dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    termination_grace_seconds: float = 5.0
    env: Dict[str, str] = field(default_factory=lambda: {
        k: v for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH")
    })

    def validate(self) -> bool:
        if not self.workdir.is_dir():
            raise ValueError(f"Subprocess workdir does not exist: {self.workdir}")
        if not self.trigger_command:
            raise ValueError("treatment_command or trigger_command is required")
        if len({Path(name).name for name in self.output_files}) != len(self.output_files):
            raise ValueError("output_files must have distinct filenames")
        grace = self.termination_grace_seconds
        if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not math.isfinite(grace) or grace <= 0:
            raise ValueError("termination_grace_seconds must be a positive finite number")
        self.parameter_args()
        return True

    def output_paths(self):
        return [Path(name) if Path(name).is_absolute() else self.workdir / name for name in self.output_files]

    def parameter_args(self):
        args = []
        for key, value in self.treatment_parameters.items():
            if not isinstance(key, str) or not key or key.startswith('-') or any(c.isspace() or c == '=' for c in key):
                raise ValueError(f"Invalid treatment parameter name: {key!r}")
            if value is None:
                args.append(f"--{key}")
            elif isinstance(value, (str, int, float, bool)):
                args.append(f"--{key}={value}")
            else:
                raise ValueError(f"Treatment parameter {key} must be scalar or null")
        return args

    @staticmethod
    def from_dict(config_yml: dict) -> "SubprocessRunnerConfig":
        config = {}
        treatment = config_yml.get("treatment_command")
        legacy = config_yml.get("trigger_command")
        if treatment and legacy and treatment != legacy:
            raise ValueError("Specify treatment_command or trigger_command, not conflicting commands")
        config["trigger_command"] = treatment or legacy
        config["treatment_parameters"] = config_yml.get("treatment_parameters") or {}
        config["output_files"] = config_yml.get("output_files") or []
        config["setup_command"] = config_yml.get("setup_command")
        config["teardown_command"] = config_yml.get("teardown_command")
        config["workdir"] = Path(config_yml["workdir"])
        config["termination_grace_seconds"] = config_yml.get("termination_grace_seconds", 5.0)
        return SubprocessRunnerConfig(**config)
