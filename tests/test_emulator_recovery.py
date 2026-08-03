import unittest
from unittest.mock import patch

from module.alas import AzurLaneAutoScript
from module.device.method import nemu_ipc
from module.exception import EmulatorNotRunningError


class FailingNemuIpc:
    def __init__(self):
        self.reconnect_calls = 0

    def reconnect(self):
        self.reconnect_calls += 1

    @nemu_ipc.retry
    def screenshot(self):
        raise nemu_ipc.NemuIpcError('emulator unavailable')


class RecoverableDevice:
    def __init__(self, restart_result=True):
        self.restart_result = restart_result
        self.release_calls = 0
        self.restart_calls = 0

    def screenshot(self):
        raise EmulatorNotRunningError('emulator unavailable')

    def nemu_ipc_release(self):
        self.release_calls += 1

    def emulator_start(self):
        self.restart_calls += 1
        return self.restart_result


class RecordingConfig:
    def __init__(self):
        self.called_tasks = []

    def task_call(self, task):
        self.called_tasks.append(task)


class TestNemuIpcRecovery(unittest.TestCase):
    def test_exhausted_nemu_ipc_retries_report_emulator_offline(self):
        client = FailingNemuIpc()

        with patch.object(nemu_ipc, 'RETRY_TRIES', 2), patch.object(nemu_ipc.time, 'sleep'):
            with self.assertRaises(EmulatorNotRunningError) as raised:
                client.screenshot()

        self.assertIsInstance(raised.exception.__cause__, nemu_ipc.NemuIpcError)
        self.assertEqual(client.reconnect_calls, 1)

    def test_run_restarts_emulator_rebuilds_device_and_schedules_login(self):
        script = AzurLaneAutoScript(config_name='test')
        device = RecoverableDevice()
        config = RecordingConfig()
        script.__dict__['device'] = device
        script.__dict__['config'] = config

        success = script.run('unused')

        self.assertFalse(success)
        self.assertEqual(device.release_calls, 1)
        self.assertEqual(device.restart_calls, 1)
        self.assertNotIn('device', script.__dict__)
        self.assertEqual(config.called_tasks, ['Restart'])

    def test_failed_emulator_restart_rebuilds_device_without_scheduling_login(self):
        script = AzurLaneAutoScript(config_name='test')
        device = RecoverableDevice(restart_result=False)
        config = RecordingConfig()
        script.__dict__['device'] = device
        script.__dict__['config'] = config

        success = script.run('unused')

        self.assertFalse(success)
        self.assertEqual(device.release_calls, 1)
        self.assertEqual(device.restart_calls, 1)
        self.assertNotIn('device', script.__dict__)
        self.assertEqual(config.called_tasks, [])


if __name__ == '__main__':
    unittest.main()
