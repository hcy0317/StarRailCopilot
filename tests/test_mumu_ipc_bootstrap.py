import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from module.device.connection import Connection
from module.device.device import Device
from module.device.method import nemu_ipc
from module.device.platform.emulator_windows import EmulatorInstance
from module.device.screenshot import Screenshot
from module.exception import EmulatorNotRunningError, RequestHumanTakeover


class TestMuMuIpcBootstrap(unittest.TestCase):
    @staticmethod
    def instance(name='MuMuPlayer-15.0-0'):
        return EmulatorInstance(
            serial='127.0.0.1:16384', name=name,
            path='C:/Emulators/MuMu/nx_main/MuMuNxMain.exe',
        )

    def test_instance_reports_android_version_for_supported_names(self):
        for name, version in (
            ('MuMuPlayer-12.0-0', '12.0'), ('MuMuPlayer-15.0-0', '15.0'),
            ('MuMuPlayerGlobal-15.0-1', '15.0'), ('YXArkNights-12.0-1', '12.0'),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.instance(name).MuMuPlayer12_android_version, version)

    def test_android_15_loads_matching_sdk_when_android_12_sdk_also_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            for version in ('12.0', '15.0'):
                dll = Path(folder, 'nx_device', version, 'shell', 'sdk', 'external_renderer_ipc.dll')
                dll.parent.mkdir(parents=True)
                dll.touch()
            expected = os.path.abspath(Path(folder, 'nx_device', '15.0', 'shell', 'sdk', 'external_renderer_ipc.dll'))

            with patch.object(nemu_ipc.ctypes, 'CDLL') as load:
                nemu_ipc.NemuIpcImpl(folder, 0, android_version='15.0')

            load.assert_called_once_with(expected)

    def test_default_android_12_sdk_selection_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            dll = Path(folder, 'nx_device', '12.0', 'shell', 'sdk', 'external_renderer_ipc.dll')
            dll.parent.mkdir(parents=True)
            dll.touch()

            with patch.object(nemu_ipc.ctypes, 'CDLL') as load:
                nemu_ipc.NemuIpcImpl(folder, 0)

            load.assert_called_once_with(os.path.abspath(dll))

    def test_legacy_shared_sdk_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            dll = Path(folder, 'shell', 'sdk', 'external_renderer_ipc.dll')
            dll.parent.mkdir(parents=True)
            dll.touch()

            with patch.object(nemu_ipc.ctypes, 'CDLL') as load:
                nemu_ipc.NemuIpcImpl(folder, 0)

            load.assert_called_once_with(os.path.abspath(dll))

    def test_android_15_does_not_load_an_android_12_only_sdk(self):
        with tempfile.TemporaryDirectory() as folder:
            dll = Path(folder, 'nx_device', '12.0', 'shell', 'sdk', 'external_renderer_ipc.dll')
            dll.parent.mkdir(parents=True)
            dll.touch()

            with patch.object(nemu_ipc.ctypes, 'CDLL') as load:
                with self.assertRaises(nemu_ipc.NemuIpcIncompatible):
                    nemu_ipc.NemuIpcImpl(folder, 0, android_version='15.0')

            load.assert_not_called()

    def test_configured_path_uses_discovered_instance_android_version(self):
        instance = self.instance()
        device = SimpleNamespace(
            config=SimpleNamespace(EmulatorInfo_path=instance.path),
            serial=instance.serial, emulator_instance=instance,
        )
        with patch.object(nemu_ipc, 'NemuIpcImpl') as factory:
            result = nemu_ipc.NemuIpc.nemu_ipc.func(device)

        self.assertIs(result, factory.return_value.__enter__.return_value)
        self.assertEqual(factory.call_args.kwargs.get('android_version'), '15.0')

    def test_discovery_fallback_uses_the_same_android_version(self):
        instance = self.instance()
        device = SimpleNamespace(
            config=SimpleNamespace(EmulatorInfo_path=instance.path),
            serial=instance.serial, emulator_instance=instance,
        )
        configured = Mock()
        configured.__enter__ = Mock(side_effect=nemu_ipc.NemuIpcError('not ready'))
        discovered = Mock()
        with patch.object(nemu_ipc, 'NemuIpcImpl', side_effect=[configured, discovered]) as factory:
            result = nemu_ipc.NemuIpc.nemu_ipc.func(device)

        self.assertIs(result, discovered)
        self.assertEqual([call.kwargs.get('android_version') for call in factory.call_args_list], ['15.0', '15.0'])
        discovered.connect_with_retry.assert_called_once_with()

    def test_initial_ipc_connection_failure_is_reported_as_unavailable(self):
        instance = self.instance()
        device = SimpleNamespace(
            config=SimpleNamespace(EmulatorInfo_path=''),
            serial=instance.serial, emulator_instance=instance,
        )
        with patch.object(nemu_ipc, 'NemuIpcImpl') as factory:
            factory.return_value.connect_with_retry.side_effect = EmulatorNotRunningError('IPC not ready')
            with self.assertRaises(RequestHumanTakeover):
                nemu_ipc.NemuIpc.nemu_ipc.func(device)

    def test_device_initialization_and_adb_screenshot_survive_unavailable_ipc(self):
        config = SimpleNamespace(
            EmulatorInfo_path='', EmulatorInfo_Emulator='MuMuPlayer12',
            Emulator_ScreenshotMethod='ADB_nc', Optimization_ScreenshotInterval=0.2,
            is_template_config=True, is_actual_task=False, Error_SaveError=False,
        )
        instance = self.instance()

        def connected(device, config):
            device.__dict__.update(config=config, serial=instance.serial, emulator_instance=instance)

        with patch.object(Connection, '__init__', connected), \
                patch.object(Device, 'is_mumu_family', True), \
                patch.object(Device, 'nemud_player_version', ''), \
                patch.object(Device, 'ldopengl_available', return_value=False), \
                patch.object(Device, 'method_check'), \
                patch.object(nemu_ipc, 'NemuIpcImpl') as factory:
            factory.return_value.connect_with_retry.side_effect = EmulatorNotRunningError('IPC not ready')
            device = Device(config=config)
            self.assertEqual(device.screenshot_method_override, '')
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            capture = Mock(return_value=image)
            device.screenshot_methods = {'ADB_nc': capture}
            device._screenshot_interval = Mock()
            device._handle_orientated_image = lambda captured: captured
            device.check_screen_size = Mock(return_value=True)
            device.check_screen_black = Mock(return_value=True)

            self.assertIs(Screenshot.screenshot(device), image)
            capture.assert_called_once_with()

    def test_keep_alive_check_uses_the_actual_android_15_instance_config(self):
        instance = self.instance()
        device = SimpleNamespace(
            config=SimpleNamespace(EmulatorInfo_path=instance.path), serial=instance.serial,
            emulator_instance=instance, is_mumu_over_version_400=True,
            check_mumu_app_keep_alive_400=Mock(return_value=True),
        )

        self.assertTrue(nemu_ipc.NemuIpc.check_mumu_app_keep_alive(device))

        device.check_mumu_app_keep_alive_400.assert_called_once_with(instance.mumu_vms_config('customer_config.json'))


if __name__ == '__main__':
    unittest.main()
