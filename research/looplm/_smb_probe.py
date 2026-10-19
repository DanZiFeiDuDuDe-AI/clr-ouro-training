# -*- coding: utf-8 -*-
"""Temporary read-only SMB access check for cheems D$ share."""
import ctypes
from ctypes import wintypes
from pathlib import Path

from _remote_cmd import USER, PASSWORD


class NETRESOURCEW(ctypes.Structure):
    _fields_ = [
        ("dwScope", wintypes.DWORD), ("dwType", wintypes.DWORD),
        ("dwDisplayType", wintypes.DWORD), ("dwUsage", wintypes.DWORD),
        ("lpLocalName", wintypes.LPWSTR), ("lpRemoteName", wintypes.LPWSTR),
        ("lpComment", wintypes.LPWSTR), ("lpProvider", wintypes.LPWSTR),
    ]


attempts = [
    ("\\\\cheems\\D$", "cheems\\" + USER),
    ("\\\\100.104.57.18\\D$", "cheems\\" + USER),
    ("\\\\cheems.tail10d158.ts.net\\D$", "cheems\\" + USER),
    ("\\\\cheems\\D$", USER),
]
connected = None
for remote, username in attempts:
    nr = NETRESOURCEW(0, 1, 0, 0, None, remote, None, None)
    rc = ctypes.windll.mpr.WNetAddConnection2W(ctypes.byref(nr), PASSWORD, username, 0)
    print("WNetAddConnection2", remote, username, "rc", rc, ctypes.FormatError(rc).strip())
    if rc == 0:
        connected = remote
        break
if connected:
    targets = [
        Path(connected + "\\training\\checkpoints\\clr_v24_o3b_h_entity_carrier_diag\\token_layer_features.npz"),
        Path(connected + "\\training\\checkpoints\\clr_v24_o21a_mr\\cache"),
        Path(connected + "\\training\\checkpoints\\clr_v24_o3b_k_first_linear_subspace\\v24_o3b_k_first_linear_subspace_report.md"),
    ]
    for p in targets:
        try:
            if p.is_dir(): print("DIR", p, "items", len(list(p.iterdir())))
            else: print("FILE", p, "bytes", p.stat().st_size)
        except Exception as ex: print("ERROR", p, repr(ex))
    ctypes.windll.mpr.WNetCancelConnection2W(connected, 0, True)
