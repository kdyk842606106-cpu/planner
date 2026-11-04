"""Fresh-process isolation and Windows/Linux resource observation."""
from __future__ import annotations

import ctypes
import os
import time


def pin_one_cpu():
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError(ctypes.get_last_error(), "Cannot inspect process affinity")
        selected = process_mask.value & -process_mask.value
        if not kernel.SetProcessAffinityMask(handle, selected):
            raise OSError(ctypes.get_last_error(), "Cannot enforce single-CPU affinity")
        return {"affinity_mask": selected, "logical_cpu": selected.bit_length() - 1}
    if hasattr(os, "sched_getaffinity"):
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
        return {"logical_cpu": selected}
    raise RuntimeError("Single-CPU affinity is unsupported on this platform")


def process_memory(pid):
    """OS peak working set includes imports; never Python-only allocation size."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        *[(field, ctypes.c_size_t) for field in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                               "PagefileUsage", "PeakPagefileUsage")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            data = Counters()
            data.cb = ctypes.sizeof(data)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(data), data.cb):
                return {"rss_bytes": data.WorkingSetSize, "peak_working_set_bytes": data.PeakWorkingSetSize}
        finally:
            kernel.CloseHandle(handle)
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as stream:
            fields = dict(line.split(":", 1) for line in stream if ":" in line)
        return {"rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
                "peak_working_set_bytes": int(fields["VmHWM"].split()[0]) * 1024}
    except (OSError, KeyError):
        return None
