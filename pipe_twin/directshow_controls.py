"""Windows IAMCameraControl without a compiler, COM package, or capture graph.

The caller owns the camera lease. Enumeration uses the same video-input
category/order as OpenCV DirectShow; this object never starts a video stream.
All interfaces stay on the constructing thread and are released on exit.
"""
from __future__ import annotations

import ctypes as ct
import os
import uuid
from typing import Any


class DirectShowControlError(OSError):
    """A native camera control could not be queried or applied."""


class _GUID(ct.Structure):
    _fields_ = [("data", ct.c_ubyte * 16)]

    @classmethod
    def parse(cls, value: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


def _check(hr: int, operation: str) -> None:
    if hr < 0:
        raise DirectShowControlError(f"{operation}: HRESULT 0x{hr & 0xffffffff:08X}")


def _invoke(pointer: ct.c_void_p, slot: int, argtypes: tuple, *args: Any) -> int:
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    method = ct.WINFUNCTYPE(ct.c_int32, ct.c_void_p, *argtypes)(table[slot])
    return int(method(pointer, *args))


class DirectShowCameraControl:
    """Read/set exposure with separate value and AUTO(1)/MANUAL(2) flags."""

    def __init__(self, index: int):
        if os.name != "nt":
            raise DirectShowControlError("DirectShow is available only on Windows")
        if type(index) is not int or index < 0:
            raise ValueError("camera index must be a non-negative integer")
        self._interfaces: list[ct.c_void_p] = []
        self._uninitialize = False
        self._control = ct.c_void_p()
        self.device_path = ""
        # WinDLL preserves HRESULT values, including RPC_E_CHANGED_MODE when
        # the stream backend already initialized this thread's COM apartment.
        self._ole = ct.WinDLL("ole32")
        self._ole.CoInitializeEx.argtypes = [ct.c_void_p, ct.c_uint32]
        self._ole.CoInitializeEx.restype = ct.c_int32
        self._ole.CoCreateInstance.argtypes = [ct.POINTER(_GUID), ct.c_void_p, ct.c_uint32,
                                              ct.POINTER(_GUID), ct.POINTER(ct.c_void_p)]
        self._ole.CoCreateInstance.restype = ct.c_int32
        self._ole.CoTaskMemFree.argtypes = [ct.c_void_p]
        self._ole.CoTaskMemFree.restype = None
        self._ole.CoUninitialize.argtypes = []
        self._ole.CoUninitialize.restype = None
        try:
            hr = self._ole.CoInitializeEx(None, 0)
            if hr >= 0:
                self._uninitialize = True
            elif hr != -2147417850:  # RPC_E_CHANGED_MODE: use existing apartment.
                _check(hr, "CoInitializeEx")
            device_enum = ct.c_void_p()
            clsid = _GUID.parse("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")
            iid = _GUID.parse("29840822-5B84-11D0-BD3B-00A0C911CE86")
            _check(self._ole.CoCreateInstance(ct.byref(clsid), None, 1, ct.byref(iid), ct.byref(device_enum)), "device enumeration")
            self._interfaces.append(device_enum)
            category = _GUID.parse("860BB310-5D01-11D0-BD3B-00A0C911CE86")
            enumerator = ct.c_void_p()
            hr = _invoke(device_enum, 3, (ct.POINTER(_GUID), ct.POINTER(ct.c_void_p), ct.c_uint32),
                         ct.byref(category), ct.byref(enumerator), 0)
            _check(hr, "CreateClassEnumerator")
            if hr != 0 or not enumerator:
                raise DirectShowControlError("No DirectShow camera devices")
            self._interfaces.append(enumerator)
            moniker = ct.c_void_p()
            for position in range(index + 1):
                moniker = ct.c_void_p()
                fetched = ct.c_uint32()
                hr = _invoke(enumerator, 3, (ct.c_uint32, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32)),
                             1, ct.byref(moniker), ct.byref(fetched))
                _check(hr, "IEnumMoniker.Next")
                if hr != 0 or fetched.value != 1:
                    raise DirectShowControlError(f"DirectShow camera index {index} unavailable")
                if position < index:
                    _invoke(moniker, 2, ())
                else:
                    self._interfaces.append(moniker)
            display = ct.c_void_p()
            hr = _invoke(moniker, 20, (ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p)), None, None, ct.byref(display))
            if hr >= 0 and display:
                try:
                    self.device_path = ct.wstring_at(display)
                finally:
                    self._ole.CoTaskMemFree(display)
            filter_pointer = ct.c_void_p()
            base_filter = _GUID.parse("56A86895-0AD4-11CE-B03A-0020AF0BA770")
            _check(_invoke(moniker, 8, (ct.c_void_p, ct.c_void_p, ct.POINTER(_GUID), ct.POINTER(ct.c_void_p)),
                           None, None, ct.byref(base_filter), ct.byref(filter_pointer)), "BindToObject")
            self._interfaces.append(filter_pointer)
            control_iid = _GUID.parse("C6E13370-30AC-11D0-A18C-00A0C9118956")
            _check(_invoke(filter_pointer, 0, (ct.POINTER(_GUID), ct.POINTER(ct.c_void_p)),
                           ct.byref(control_iid), ct.byref(self._control)), "IAMCameraControl")
            self._interfaces.append(self._control)
        except BaseException:
            self.close()
            raise

    def exposure_range(self) -> dict[str, int]:
        values = [ct.c_int32() for _ in range(5)]
        _check(_invoke(self._control, 3, (ct.c_int32,) + (ct.POINTER(ct.c_int32),)*5,
                       4, *(ct.byref(value) for value in values)), "Exposure.GetRange")
        return dict(zip(("minimum", "maximum", "step", "default", "capabilities"), (v.value for v in values)))

    def exposure(self) -> dict[str, int]:
        value, flags = ct.c_int32(), ct.c_int32()
        _check(_invoke(self._control, 5, (ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)),
                       4, ct.byref(value), ct.byref(flags)), "Exposure.Get")
        return {"value": value.value, "flags": flags.value}

    def set_exposure(self, value: int, flags: int) -> None:
        if flags not in (1, 2):
            raise ValueError("Exposure requires exactly AUTO(1) or MANUAL(2)")
        _check(_invoke(self._control, 4, (ct.c_int32, ct.c_int32, ct.c_int32), 4, value, flags), "Exposure.Set")

    def close(self) -> None:
        for pointer in reversed(self._interfaces):
            _invoke(pointer, 2, ())
        self._interfaces.clear()
        if self._uninitialize:
            self._ole.CoUninitialize()
            self._uninitialize = False

    def __enter__(self) -> "DirectShowCameraControl":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
