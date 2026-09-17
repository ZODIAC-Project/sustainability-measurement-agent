import logging
import json
import os
import subprocess
import shlex
import shutil

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Dict, Optional
import threading

from sma.model import SMAObserver, Triggerable, Triggerable

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class SubprocessrunnerSmaModule(SMAObserver, Triggerable):
    """
    SMA Module that runs a subprocess locally as treatment. This works well when SMA is 
    running inside the cluster, e.g. as a job, and for treatments that are not easily
    bottlenecked by noise."""

    def __init__(self, config: dict):
        self.config: SubprocessRunnerConfig = SubprocessRunnerConfig.from_dict(config)
        log.info(json.dumps(config, indent=4))
        if not self.config.validate():
            raise ValueError("Invalid SubprocessRunner configuration.")
        log.debug(f"Loaded SubprocessRunner configuration: {self.config}")

    # --- HOOKS

    def trigger(self, cancel: threading.Event, **kwargs) -> Optional[Dict[str, Any]]:
        self.cancel = cancel
        self._output_before = {str(path): self._signature(path) for path in self.config.output_paths()}
        command = shlex.join(shlex.split(self.config.trigger_command) + self.config.parameter_args())
        exit_code = self._run_subprocess(command)
        return {"subprocess_exit_code": exit_code}

    def onSetup(self):
        if self.config.setup_command:
            self._run_subprocess(self.config.setup_command)

    def onTeardown(self):
        if self.config.teardown_command:
            self._run_subprocess(self.config.teardown_command)



    @staticmethod
    def _signature(path):
        if not path.is_file():
            return None
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def onReport(self, report=None):
        if report is None:
            return
        destination = Path(report.location) / "subprocess_output"
        for source in self.config.output_paths():
            signature = self._signature(source)
            if signature is None or signature == getattr(self, "_output_before", {}).get(str(source)):
                log.warning(f"No new output file for this run: {source}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)

    # --- PRIVATE METHODS

    def _run_subprocess(self, command: str, cancel: Optional[threading.Event] = None) -> int:
        log.info(f"Running subprocess command: {command}")

        # todo: cancel

        try:
            result = subprocess.run(
                shlex.split(command), 
                cwd=self.config.workdir, 
                env = self.config.env,
                capture_output=True, 
                text=True
                )
            log.debug(f"Subprocess finished with return code {result.returncode}")
            if result.stdout:
                log.debug(f"Subprocess stdout: {result.stdout}")
            if result.stderr:
                log.error(f"Subprocess stderr: {result.stderr}")
            return result.returncode
        except Exception as e:
            log.error(f"Error running subprocess: {e}")
            return -1

@dataclass
class SubprocessRunnerConfig:
    trigger_command: str
    setup_command: str
    teardown_command: str
    workdir: Path
    treatment_parameters: Dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    env = {k: v for k, v in os.environ.items() # todo
       if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH")}

    def validate(self) -> bool:
        if not self.workdir.is_dir():
            raise ValueError(f"Subprocess workdir does not exist: {self.workdir}")
        if not self.trigger_command:
            raise ValueError("treatment_command or trigger_command is required")
        if len({Path(name).name for name in self.output_files}) != len(self.output_files):
            raise ValueError("output_files must have distinct filenames")
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
        return SubprocessRunnerConfig(**config)
