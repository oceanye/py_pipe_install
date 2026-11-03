"""Describe implemented computation and reachable addresses without starting services."""
from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess


def probe_capabilities() -> dict:
    """Hardware availability does not imply an implemented GPU analysis path."""
    import cv2

    count = 0
    probe_error = None
    cuda = getattr(cv2, 'cuda', None)
    try:
        count = max(0, int(cuda.getCudaEnabledDeviceCount())) if cuda else 0
    except (AttributeError, RuntimeError, cv2.error):
        probe_error = 'CUDA_RUNTIME_UNAVAILABLE'
    stereo_api = bool(cuda and hasattr(cuda, 'createStereoSGM'))
    reason = ('CUDA_DETECTED_BACKEND_NOT_IMPLEMENTED' if count and stereo_api
              else 'CUDA_STEREO_UNAVAILABLE')
    return {
        'backend': 'CPU', 'execution_backend': 'CPU', 'supported_backends': ['CPU'],
        'opencv_version': cv2.__version__,
        'preprocessing': {
            'fast_nl_means_available': callable(getattr(cv2, 'fastNlMeansDenoising', None)),
            'bilateral_available': callable(getattr(cv2, 'bilateralFilter', None)),
            'clahe_available': callable(getattr(cv2, 'createCLAHE', None)),
        },
        'cpu': {'logical_cores': os.cpu_count() or 1, 'opencv_threads': cv2.getNumThreads()},
        'gpu': {'cuda_device_count': count, 'cuda_stereo_api_available': stereo_api,
                'execution_supported': False, 'reason': reason, 'probe_error': probe_error},
        'note': '当前算法使用 CPU；CUDA 设备检测不代表已启用 GPU 解算。',
    }


def listening_addresses() -> list[str]:
    """Candidate IPv4 bind addresses; never change interfaces or firewall rules."""
    candidates = set()
    try:
        candidates.update(item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM))
    except OSError:
        pass
    # Hostname lookup on Windows may omit the Tailscale adapter. Its CLI only
    # reads current addresses; no login, network configuration or installation.
    command = shutil.which('tailscale')
    if command:
        try:
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            response = subprocess.run([command, 'ip', '-4'], capture_output=True,
                                      text=True, timeout=2, check=False, **options)
            if response.returncode == 0:
                candidates.update(response.stdout.split())
        except (OSError, subprocess.TimeoutExpired):
            pass
    addresses = []
    tailscale = ipaddress.ip_network('100.64.0.0/10')
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if (address.version == 4 and not (address.is_unspecified or address.is_multicast
                or address.is_link_local or address.is_loopback)
                and (address.is_private or address in tailscale)):
            addresses.append(str(address))
    # Prefer the VPN where available, then a LAN address. Loopback is always
    # offered for a same-machine end-to-end check but not advertised as remote.
    addresses.sort(key=lambda value: (ipaddress.ip_address(value) not in tailscale,
                                      int(ipaddress.ip_address(value))))
    return addresses + ['127.0.0.1']
