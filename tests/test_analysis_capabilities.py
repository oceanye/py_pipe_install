import subprocess

from pipe_twin import analysis_capabilities as capabilities


def test_cuda_hardware_does_not_advertise_an_unimplemented_execution_backend(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 2)
    monkeypatch.setattr(cv2.cuda, 'createStereoSGM', object(), raising=False)
    result = capabilities.probe_capabilities()
    assert result['gpu']['cuda_device_count'] == 2
    assert result['gpu']['cuda_stereo_api_available']
    assert not result['gpu']['execution_supported']
    assert result['execution_backend'] == 'CPU'
    assert result['supported_backends'] == ['CPU']


def test_no_cuda_environment_remains_a_usable_cpu_server(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2.cuda, 'getCudaEnabledDeviceCount', lambda: 0)
    result = capabilities.probe_capabilities()
    assert result['backend'] == 'CPU'
    assert result['cpu']['logical_cores'] >= 1
    assert result['gpu']['reason'] == 'CUDA_STEREO_UNAVAILABLE'


def test_addresses_prefer_vpn_and_do_not_offer_wildcard_public_or_link_local(monkeypatch):
    monkeypatch.setattr(capabilities.socket, 'getaddrinfo', lambda *_: [
        (None, None, None, None, (ip, 0)) for ip in
        ('192.168.1.3', '0.0.0.0', '8.8.8.8', '169.254.1.8', '127.0.0.1')])
    monkeypatch.setattr(capabilities.shutil, 'which', lambda _: 'tailscale')
    monkeypatch.setattr(capabilities.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, '100.80.2.3\n'))
    assert capabilities.listening_addresses() == ['100.80.2.3', '192.168.1.3', '127.0.0.1']


def test_address_discovery_can_fall_back_without_network_configuration(monkeypatch):
    def fail(*_):
        raise OSError('no DNS')
    monkeypatch.setattr(capabilities.socket, 'getaddrinfo', fail)
    monkeypatch.setattr(capabilities.shutil, 'which', lambda _: None)
    assert capabilities.listening_addresses() == ['127.0.0.1']
