"""Local process regressions: no external services or cloud credentials required."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml
from click.testing import CliRunner

from cli.main import cli
from modules.SubprocessRunner.main import SubprocessrunnerSmaModule
from sma import Config, ReportIO, SustainabilityMeasurementAgent


def command(script, source):
    script.write_text(source)
    return shlex.join([sys.executable, '-u', str(script)])


def config_file(tmp_path, treatment, *, max_duration=None, **module_options):
    module = {'workdir': str(tmp_path), 'treatment_command': treatment,
              'termination_grace_seconds': 0.15, **module_options}
    observation = {'mode': 'module', 'module_trigger': 'child'}
    if max_duration is not None:
        observation['max_duration'] = max_duration
    content = {'sma': {
        'modules': {'child': {'module': 'SubprocessRunner', 'config': module}},
        'observation': observation,
        'report': {'location': 'reports/${runHash}'},
    }}
    config = tmp_path / 'sma.yaml'
    config.write_text(yaml.safe_dump(content))
    return config


def only_report(tmp_path):
    reports = list((tmp_path / 'reports').glob('*/run.json'))
    assert len(reports) == 1
    return reports[0].parent, json.loads(reports[0].read_text())


@pytest.mark.parametrize('max_duration', [None, 5])
def test_nonzero_exit_reaches_cli_and_report_with_partial_artifacts(tmp_path, monkeypatch, max_duration):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'fail.py', '''import sys
from pathlib import Path
print('partial stdout')
print('partial stderr', file=sys.stderr)
Path('result.json').write_text('{"status":"ERROR"}')
sys.exit(7)
''')
    config = config_file(tmp_path, treatment, max_duration=max_duration, output_files=['result.json'])
    result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
    assert result.exit_code == 7, result.output
    directory, run = only_report(tmp_path)
    assert run['status'] == 'error'
    assert run['user_data']['subprocess_exit_code'] == 7
    assert (directory / 'subprocess_output/result.json').read_text() == '{"status":"ERROR"}'
    assert (directory / 'subprocess_logs/treatment.stdout.log').read_text() == 'partial stdout\n'
    assert (directory / 'subprocess_logs/treatment.stderr.log').read_text() == 'partial stderr\n'
    restored = ReportIO.load_from_location(str(directory.relative_to(tmp_path)), Config.from_file(config))
    assert restored.run_data.status == 'error'
    assert restored.run_data.user_data['subprocess_exit_code'] == 7


def test_success_preserves_large_binary_streams_and_quoted_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'child with spaces.py', '''import os, sys
from pathlib import Path
assert set(sys.argv[1:]) == {'--label=a b;$(false)', '--enabled'}
os.write(1, b'x' * 200000 + b'\\xff')
os.write(2, b'y' * 200000)
Path('fresh.json').write_text('{}')
''')
    (tmp_path / 'stale.json').write_text('stale')
    config = config_file(tmp_path, treatment, max_duration=5,
                         treatment_parameters={'label': 'a b;$(false)', 'enabled': None},
                         output_files=['fresh.json', 'stale.json'])
    result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
    assert result.exit_code == 0, result.output
    directory, run = only_report(tmp_path)
    assert run['status'] == 'ok'
    assert run['user_data']['subprocess_exit_code'] == 0
    assert (directory / 'subprocess_logs/treatment.stdout.log').read_bytes() == b'x' * 200000 + b'\xff'
    assert (directory / 'subprocess_logs/treatment.stderr.log').stat().st_size == 200000
    assert (directory / 'subprocess_output/fresh.json').is_file()
    assert not (directory / 'subprocess_output/stale.json').exists()


@pytest.mark.parametrize('phase,exit_code', [('setup', 3), ('teardown', 9)])
def test_lifecycle_failure_is_reported_and_teardown_logs_survive(tmp_path, monkeypatch, phase, exit_code):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'treatment.py', "from pathlib import Path; Path('treated').touch()")
    setup = command(tmp_path / 'setup.py', "print('setup output'); raise SystemExit(%d)" % (exit_code if phase == 'setup' else 0))
    teardown = command(tmp_path / 'teardown.py', "print('teardown output'); raise SystemExit(%d)" % (exit_code if phase == 'teardown' else 0))
    config = config_file(tmp_path, treatment, setup_command=setup, teardown_command=teardown)
    result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
    assert result.exit_code == 1, result.output
    directory, run = only_report(tmp_path)
    assert run['status'] == 'error'
    assert run['user_data'][f'subprocess_{phase}_exit_code'] == exit_code
    assert run['user_data']['lifecycle_errors'][0]['event'] == ('onSetup' if phase == 'setup' else 'onTeardown')
    assert (directory / 'subprocess_logs/setup.stdout.log').read_text() == 'setup output\n'
    assert (directory / 'subprocess_logs/teardown.stdout.log').read_text() == 'teardown output\n'
    assert (tmp_path / 'treated').exists() == (phase == 'teardown')
    if phase == 'setup':
        assert 'subprocess_exit_code' not in run['user_data']
    else:
        assert run['user_data']['subprocess_exit_code'] == 0


def process_running(pid):
    # Orphans may remain zombies until the host's PID 1 reaps them; they cannot run.
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        return stat.read_text().split(') ', 1)[1].split()[0] != 'Z'
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process group and /proc assertion')
@pytest.mark.parametrize('parent_ignores_term', [False, True])
def test_deadline_kills_child_and_grandchild_and_preserves_timeout_metadata(tmp_path, monkeypatch, parent_ignores_term):
    monkeypatch.chdir(tmp_path)
    child = command(tmp_path / 'grandchild.py', '''import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('grandchild.pid').write_text(str(os.getpid()))
while True: time.sleep(0.1)
''')
    parent_source = '''import os, signal, subprocess, time, sys
from pathlib import Path
signal.signal(signal.SIGTERM, %s)
subprocess.Popen(%r)
Path('parent.pid').write_text(str(os.getpid()))
print('started parent')
print('stderr before timeout', file=sys.stderr)
while True: time.sleep(0.1)
''' % ('signal.SIG_IGN' if parent_ignores_term else 'lambda *_: sys.exit(0)', shlex.split(child))
    treatment = command(tmp_path / 'parent.py', parent_source)
    config = config_file(tmp_path, treatment, max_duration=1, output_files=['parent.pid', 'grandchild.pid'])
    started = time.monotonic()
    try:
        result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
        assert result.exit_code == 124, result.output
        assert time.monotonic() - started < 5
        directory, run = only_report(tmp_path)
        assert run['status'] == 'timeout'
        assert run['user_data']['subprocess_cancelled'] is True
        assert run['user_data']['subprocess_exit_code'] == (-signal.SIGKILL if parent_ignores_term else 0)
        assert (directory / 'subprocess_logs/treatment.stdout.log').read_text() == 'started parent\n'
        assert (directory / 'subprocess_logs/treatment.stderr.log').read_text() == 'stderr before timeout\n'
        for filename in ('parent.pid', 'grandchild.pid'):
            pid = int((tmp_path / filename).read_text())
            assert not process_running(pid), f'{filename} process {pid} survived cancellation'
            assert (directory / 'subprocess_output' / filename).exists()
    finally:
        for filename in ('parent.pid', 'grandchild.pid'):
            file = tmp_path / filename
            if file.exists():
                try:
                    os.kill(int(file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_missing_executable_is_error_not_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = config_file(tmp_path, str(tmp_path / 'missing-command'), max_duration=5)
    result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
    assert result.exit_code == 1, result.output
    directory, run = only_report(tmp_path)
    assert run['status'] == 'error'
    assert 'No such file' in run['user_data']['error']
    assert 'subprocess_exit_code' not in run['user_data']
    assert (directory / 'subprocess_logs/treatment.stderr.log').is_file()


def test_api_run_nonzero_and_late_observer_failure_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'fail.py', 'raise SystemExit(6)')
    config = Config.from_file(config_file(tmp_path, treatment))

    class BadObserver:
        def onRunEnd(self, run=None):
            raise RuntimeError('run end failed')

    agent = SustainabilityMeasurementAgent(config, observers=[BadObserver()])
    with agent.start_session():
        agent.run(subprocess_exit_code=0)  # Annotations cannot mask the real exit.
    _, run = only_report(tmp_path)
    assert agent.last_run.status == run['status'] == 'error'
    assert run['user_data']['subprocess_exit_code'] == 6
    assert run['user_data']['lifecycle_errors'][0]['event'] == 'onRunEnd'


def test_legacy_report_without_status_can_still_be_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'ok.py', 'pass')
    config = config_file(tmp_path, treatment)
    assert CliRunner().invoke(cli, ['run', str(config), '--probe', 'none']).exit_code == 0
    directory, run = only_report(tmp_path)
    del run['status']
    (directory / 'run.json').write_text(json.dumps(run))
    restored = ReportIO.load_from_location(str(directory.relative_to(tmp_path)), Config.from_file(config))
    assert restored.run_data.status is None


def test_timer_duration_is_successful_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / 'timer.yaml'
    config.write_text(yaml.safe_dump({'sma': {'observation': {
        'mode': 'timer', 'max_duration': 1, 'window': {'left': 0, 'right': 0}},
        'report': {'location': 'reports/${runHash}'}}}))
    result = CliRunner().invoke(cli, ['run', str(config), '--probe', 'none'])
    assert result.exit_code == 0, result.output
    assert only_report(tmp_path)[1]['status'] == 'ok'


def test_real_cli_process_preserves_child_exit_code(tmp_path):
    treatment = command(tmp_path / 'fail.py', 'raise SystemExit(13)')
    config = config_file(tmp_path, treatment)
    result = subprocess.run([sys.executable, '-m', 'cli.main', 'run', str(config), '--probe', 'none'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 13, result.stderr
    assert only_report(tmp_path)[1]['status'] == 'error'


def test_already_cancelled_trigger_does_not_start_child(tmp_path):
    treatment = command(tmp_path / 'never.py', "from pathlib import Path; Path('started').touch()")
    runner = SubprocessrunnerSmaModule({'workdir': str(tmp_path), 'treatment_command': treatment})
    cancel = threading.Event()
    cancel.set()
    metadata = runner.trigger(cancel)
    assert metadata == {'subprocess_cancelled': True}
    assert not (tmp_path / 'started').exists()


def test_measurement_error_cannot_discard_child_report_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    treatment = command(tmp_path / 'child.py', "from pathlib import Path; Path('result.json').write_text('{}')")
    config = Config.from_file(config_file(tmp_path, treatment, output_files=['result.json']))

    class BrokenMeasurement:
        def observe(self, **kwargs):
            raise ValueError('invalid measurement response')

    monkeypatch.setattr(config, 'measurement_queries', lambda: {'energy': BrokenMeasurement()})
    agent = SustainabilityMeasurementAgent(config)
    with agent.start_session():
        agent.run()
    directory, run = only_report(tmp_path)
    assert run['status'] == 'error'
    assert run['user_data']['subprocess_exit_code'] == 0
    assert run['user_data']['measurement_errors'] == [
        {'measurement': 'energy', 'error': 'invalid measurement response'}]
    assert (directory / 'subprocess_output/result.json').read_text() == '{}'
    assert (directory / 'subprocess_logs/treatment.stdout.log').is_file()


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process group and /proc assertion')
@pytest.mark.parametrize('phase', ['setup', 'treatment', 'teardown'])
def test_external_sigterm_cancels_active_phase_and_preserves_report(tmp_path, phase):
    descendant = command(tmp_path / 'descendant.py', """import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('descendant.pid').write_text(str(os.getpid()))
while True: time.sleep(0.1)
""")
    hanging = command(tmp_path / 'hanging.py', """import os, signal, subprocess, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen(%r)
while not Path('descendant.pid').exists(): time.sleep(0.01)
Path('active.pid').write_text(str(os.getpid()))
print('active phase ready')
while True: time.sleep(0.1)
""" % shlex.split(descendant))
    noop = command(tmp_path / 'noop.py', 'pass')
    options = {f'{phase}_command': hanging} if phase != 'treatment' else {}
    config = config_file(tmp_path, hanging if phase == 'treatment' else noop,
                         max_duration=30, output_files=['active.pid', 'descendant.pid'], **options)
    content = yaml.safe_load(config.read_text())
    content['sma']['observation']['window'] = {'left': 0, 'right': 30}
    if phase == 'teardown':
        content['sma']['observation']['window']['right'] = 0
    config.write_text(yaml.safe_dump(content))
    proc = subprocess.Popen([sys.executable, '-m', 'cli.main', 'run', str(config), '--probe', 'none'],
                            cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        until = time.monotonic() + 5
        while not (tmp_path / 'active.pid').exists() and proc.poll() is None and time.monotonic() < until:
            time.sleep(0.01)
        assert (tmp_path / 'active.pid').exists(), 'SMA did not launch the phase'
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=5)
        assert proc.returncode == 124, output
        directory, run = only_report(tmp_path)
        assert run['status'] == 'timeout'
        assert run['user_data']['cancellation_requested'] is True
        key = 'subprocess_exit_code' if phase == 'treatment' else f'subprocess_{phase}_exit_code'
        assert run['user_data'][key] == -signal.SIGKILL
        assert (directory / f'subprocess_logs/{phase}.stdout.log').read_text() == 'active phase ready\n'
        for filename in ('active.pid', 'descendant.pid'):
            assert not process_running(int((tmp_path / filename).read_text()))
            assert (directory / 'subprocess_output' / filename).exists()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate(timeout=5)
        for filename in ('active.pid', 'descendant.pid'):
            path = tmp_path / filename
            if path.exists():
                try:
                    os.kill(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process group assertion')
def test_real_smew_external_cancel_waits_for_sma_child_cleanup(tmp_path, monkeypatch):
    smew = Path(__file__).resolve().parents[2] / 'SMEW'
    if not (smew / 'smew/runner.py').is_file():
        pytest.skip('optional external supervisor test needs sibling SMEW checkout')
    pytest.importorskip('pydantic')
    monkeypatch.syspath_prepend(str(smew))
    from smew.runner import execute

    hanging = command(tmp_path / 'hanging.py', """import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('active.pid').write_text(str(os.getpid()))
while True: time.sleep(0.1)
""")
    config = yaml.safe_load(config_file(tmp_path, hanging, max_duration=30).read_text())
    pidfile = tmp_path / 'active.pid'
    try:
        outcome = execute(config, workdir=tmp_path,
            command=[sys.executable, '-m', 'cli.main', 'run', '{config}', '--probe', 'none'],
            should_cancel=pidfile.exists, timeout=8, grace_period=3, poll_interval=0.02,
            env={'LOGLEVEL': 'ERROR'})
        assert outcome.state == 'cancelled', outcome.log_tail
        assert outcome.exit_code == 124, outcome.log_tail
        assert outcome.report_dir is not None
        assert not process_running(int(pidfile.read_text()))
        _, run = only_report(tmp_path)
        assert run['status'] == 'timeout'
        assert run['user_data']['subprocess_exit_code'] == -signal.SIGKILL
        assert run['user_data']['cancellation_requested'] is True
    finally:
        if pidfile.exists():
            try:
                os.killpg(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('response_kind', ['empty', 'http_error', 'api_error'])
def test_empty_optional_series_is_warning_but_query_errors_fail(tmp_path, monkeypatch, response_kind):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            status = 200
            body = {'ready': True}
            if parsed.path == '/api/v1/query_range':
                query = parse_qs(parsed.query)
                if query['query'] == ['optional']:
                    status = 503 if response_kind == 'http_error' else 200
                    body = ({'status': 'success', 'data': {'resultType': 'matrix', 'result': []}}
                            if response_kind == 'empty' else {'status': 'error', 'error': 'fixture query failed'})
                else:
                    body = {'status': 'success', 'data': {'resultType': 'matrix', 'result': [
                        {'metric': {'job': 'fixture'}, 'values': [[float(query['start'][0]), '1'],
                                                                [float(query['end'][0]), '2']]}]}}
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        monkeypatch.chdir(tmp_path)
        noop = command(tmp_path / 'noop.py', 'pass')
        path = config_file(tmp_path, noop)
        config = yaml.safe_load(path.read_text())
        config['sma']['services'] = {'prometheus': {'address': f'http://127.0.0.1:{server.server_port}'}}
        config['sma']['measurements'] = [{name: {'query': name}} for name in ('healthy', 'optional')]
        path.write_text(yaml.safe_dump(config))
        result = CliRunner().invoke(cli, ['run', str(path), '--probe', 'none'])
        assert result.exit_code == (0 if response_kind == 'empty' else 1), result.output
        directory, run = only_report(tmp_path)
        assert run['status'] == ('ok' if response_kind == 'empty' else 'error')
        key = 'measurement_warnings' if response_kind == 'empty' else 'measurement_errors'
        assert run['user_data'][key][0]['measurement'] == 'optional'
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert set(manifest['data_files']) == {'healthy'}
        assert manifest['data_files']['healthy']['rows'] == 2
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)
