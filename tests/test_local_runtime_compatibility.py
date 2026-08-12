import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from module.device.method.scrcpy.core import ScrcpyCore
from module.device.platform.emulator_windows import Emulator
from module.device.platform.platform_windows import PlatformWindows


class TestScrcpyCompatibility(unittest.TestCase):
    def test_server_command_matches_configured_jar_version(self):
        core = object.__new__(ScrcpyCore)

        for version in ('1.20', '1.25'):
            with self.subTest(version=version):
                core.config = SimpleNamespace(
                    SCRCPY_FILEPATH_LOCAL=f'./bin/scrcpy/scrcpy-server-v{version}.jar',
                    SCRCPY_FILEPATH_REMOTE=f'/data/local/tmp/scrcpy-server-v{version}.jar',
                )

                command = core._scrcpy_server_command()

                self.assertEqual(command[3], 'com.genymobile.scrcpy.Server')
                self.assertEqual(command[4], version)


class TestMuMuRuntimeCompatibility(unittest.TestCase):
    def test_reads_adb_serial_from_vm_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'vm_config.json')
            with open(path, 'w', encoding='utf-8') as file:
                json.dump({'vm': {'nat': {'port_forward': {'adb': {'host_port': '16384'}}}}}, file)

            self.assertEqual(Emulator.mumu12_vm_config_to_serial(path), '127.0.0.1:16384')

    def test_invalid_vm_config_is_ignored(self):
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as file:
            file.write('{invalid json')
            path = file.name
        try:
            self.assertEqual(Emulator.mumu12_vm_config_to_serial(path), '')
        finally:
            os.unlink(path)

    @patch('module.device.platform.platform_windows.subprocess.run')
    def test_process_state_accepts_indexed_cli_payload(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({'0': {'is_process_started': True}}),
        )

        self.assertIs(PlatformWindows._mumu12_process_started('mumu-cli.exe', 0), True)
        run.assert_called_once_with(
            ['mumu-cli.exe', 'info', '--vmindex', '0'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10,
        )

    def test_stopped_instance_skips_shutdown(self):
        platform = object.__new__(PlatformWindows)
        platform._mumu12_process_started = Mock(return_value=False)
        platform.execute = Mock()

        platform._mumu12_stop('mumu-cli.exe', 0)

        platform.execute.assert_not_called()

    @patch('module.device.platform.platform_windows.time.sleep')
    def test_running_instance_waits_for_shutdown(self, sleep):
        platform = object.__new__(PlatformWindows)
        platform._mumu12_process_started = Mock(side_effect=[True, False])
        process = Mock()
        platform.execute = Mock(return_value=process)

        platform._mumu12_stop('mumu-cli.exe', 1)

        platform.execute.assert_called_once_with('"mumu-cli.exe" control --vmindex 1 shutdown')
        process.wait.assert_called_once_with(timeout=10)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
