# -*- coding: utf-8 -*-
"""Inventory/download the frozen O3b artifacts from cheems."""
import json
import os
import posixpath
import socket
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko

HOST = "cheems"
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
REMOTE_ROOT = "D:/training"
LOCAL_ROOT = Path(r"H:/LoopLM/remote_archive/cheems")
REMOTE_PATHS = [
    "D:/training/checkpoints/clr_v24_o3b_k_first_linear_subspace",
    "D:/training/checkpoints/clr_v24_o21b/b1_best.pt",
    "D:/training/a1_v24_o3b.py",
    "D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag",
    "D:/training/checkpoints/clr_v24_o21a_mr/cache",
]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def fast_sftp():
    sock = socket.create_connection((HOST, 22), timeout=30)
    transport = paramiko.Transport(sock, default_window_size=128 * 1024 * 1024, default_max_packet_size=4 * 1024 * 1024)
    transport.connect(username=USER, password=PASSWORD)
    return transport, paramiko.SFTPClient.from_transport(transport)


def stream_file(remote, local):
    sock = socket.create_connection((HOST, 22), timeout=30)
    transport = paramiko.Transport(sock, default_window_size=128 * 1024 * 1024, default_max_packet_size=4 * 1024 * 1024)
    transport.connect(username=USER, password=PASSWORD)
    channel = transport.open_session()
    win_remote = remote.replace("/", "\\")
    channel.exec_command(f'cmd /c type "{win_remote}"')
    local = Path(local); local.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    try:
        with open(local, "wb") as out:
            while True:
                if channel.recv_ready():
                    data = channel.recv(4 * 1024 * 1024)
                    if not data: break
                    out.write(data); done += len(data)
                    if done % (64 * 1024 * 1024) < len(data): print(f"streamed {done/1024**2:.1f} MiB", flush=True)
                elif channel.exit_status_ready():
                    while channel.recv_ready():
                        data = channel.recv(4 * 1024 * 1024)
                        if not data: break
                        out.write(data); done += len(data)
                    break
                else:
                    import time; time.sleep(.05)
        rc = channel.recv_exit_status()
        err = b""
        while channel.recv_stderr_ready(): err += channel.recv_stderr(1024 * 1024)
        print(f"stream complete bytes={done} rc={rc} stderr={err.decode('utf-8','replace')[-1000:]}", flush=True)
    finally:
        channel.close(); transport.close()


def download_carrier_fast():
    remote = "D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag/token_layer_features.npz"
    local = rel_local(remote); local.parent.mkdir(parents=True, exist_ok=True)
    t, sftp = fast_sftp()
    try:
        size = sftp.stat(remote).st_size
        print(f"fast pull {remote} -> {local} ({size/1024**2:.1f} MiB)", flush=True)
        with sftp.open(remote, "rb") as src, open(local, "wb") as dst:
            src.prefetch(file_size=size, max_concurrent_requests=64)
            done = 0
            while done < size:
                block = src.read(min(16 * 1024 * 1024, size - done))
                if not block: raise IOError(f"short read at {done}")
                dst.write(block); done += len(block)
                print(f"progress {done/1024**2:.1f}/{size/1024**2:.1f} MiB", flush=True)
    finally:
        sftp.close(); t.close()


def remote_inventory(c):
    cmd = r'''powershell -NoProfile -Command "$items=@(Get-ChildItem -Recurse -File 'D:\training\checkpoints\clr_v24_o3b_k_first_linear_subspace' | Select-Object FullName,Length,LastWriteTime; Get-Item 'D:\training\checkpoints\clr_v24_o21b\b1_best.pt','D:\training\a1_v24_o3b.py' | Select-Object FullName,Length,LastWriteTime; Get-ChildItem -Recurse -File 'D:\training\checkpoints\clr_v24_o3b_h_entity_carrier_diag' | Measure-Object Length -Sum | Select-Object Count,Sum; Get-ChildItem -Recurse -File 'D:\training\checkpoints\clr_v24_o21a_mr\cache' | Measure-Object Length -Sum | Select-Object Count,Sum); $items | ConvertTo-Json -Compress"'''
    _, out, err = c.exec_command(cmd, timeout=90)
    text = out.read().decode("utf-8", "replace")
    errors = err.read().decode("utf-8", "replace")
    print(text)
    if errors: print(errors, file=sys.stderr)


def sftp_stat(sftp, path):
    for candidate in (path, "/" + path):
        try:
            return candidate, sftp.stat(candidate)
        except Exception:
            pass
    raise FileNotFoundError(path)


def walk(sftp, path):
    actual, st = sftp_stat(sftp, path)
    if st.st_mode & 0o170000 == 0o100000:
        yield actual, st.st_size
        return
    for item in sftp.listdir_attr(actual):
        child = posixpath.join(actual, item.filename)
        if item.st_mode & 0o170000 == 0o100000:
            yield child, item.st_size
        elif item.st_mode & 0o170000 == 0o040000:
            yield from walk(sftp, child)


def rel_local(remote_path):
    p = remote_path.replace("\\", "/")
    if p.startswith("/D:/"): p = p[1:]
    if p.lower().startswith(REMOTE_ROOT.lower() + "/"):
        p = p[len(REMOTE_ROOT) + 1:]
    elif p.lower().startswith(REMOTE_ROOT.lower()):
        p = p[len(REMOTE_ROOT):].lstrip("/")
    return LOCAL_ROOT / Path(*p.split("/"))


def download(c):
    sftp = c.open_sftp()
    try:
        all_files = []
        for root in REMOTE_PATHS:
            found = list(walk(sftp, root))
            print(f"{root}: {len(found)} file(s), {sum(x[1] for x in found)/1024**3:.3f} GiB", flush=True)
            all_files.extend(found)
        total = sum(size for _, size in all_files)
        done = 0
        for remote, size in all_files:
            local = rel_local(remote)
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == size:
                done += size
                print(f"skip existing {local} ({size/1024**2:.1f} MiB)", flush=True)
                continue
            print(f"pull {remote} -> {local} ({size/1024**2:.1f} MiB)", flush=True)
            sftp.get(remote, str(local))
            done += size
            print(f"progress {done/1024**3:.3f}/{total/1024**3:.3f} GiB", flush=True)
    finally:
        sftp.close()


def pull_part(remote, start, end, part_path):
    c = connect()
    sftp = c.open_sftp()
    try:
        f = sftp.open(remote, "rb")
        f.seek(start)
        remaining = end - start
        with open(part_path, "wb") as out:
            while remaining:
                block = f.read(min(8 * 1024 * 1024, remaining))
                if not block: raise IOError(f"short read at {start + (end - start - remaining)}")
                out.write(block); remaining -= len(block)
        f.close()
        return part_path
    finally:
        sftp.close(); c.close()


def download_fast(c):
    """Parallel range pull for large files over a slow relay."""
    sftp = c.open_sftp()
    try:
        files = []
        for root in REMOTE_PATHS: files.extend(list(walk(sftp, root)))
    finally:
        sftp.close()
    for remote, size in files:
        local = rel_local(remote); local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size == size:
            print(f"skip existing {local} ({size/1024**2:.1f} MiB)", flush=True); continue
        if size < 16 * 1024 * 1024:
            c2 = connect(); s2 = c2.open_sftp()
            try: print(f"pull {remote} -> {local}", flush=True); s2.get(remote, str(local))
            finally: s2.close(); c2.close()
            continue
        workers = min(8, max(2, (size + 64*1024*1024 - 1) // (64*1024*1024)))
        tmp = Path(tempfile.mkdtemp(prefix="o3bl_", dir=str(local.parent)))
        parts = []
        print(f"parallel pull {remote} -> {local}: {size/1024**2:.1f} MiB with {workers} streams", flush=True)
        try:
            ranges = []
            for i in range(workers):
                start = size * i // workers; end = size * (i + 1) // workers
                part = tmp / f"part_{i:02d}"
                ranges.append((remote, start, end, str(part)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(pull_part, *args) for args in ranges]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result(); print(f"  completed {i}/{workers}", flush=True)
            with open(local, "wb") as out:
                for _, _, _, part in ranges:
                    with open(part, "rb") as inp:
                        while True:
                            block = inp.read(8 * 1024 * 1024)
                            if not block: break
                            out.write(block)
            if local.stat().st_size != size: raise IOError(f"size mismatch {local}: {local.stat().st_size} != {size}")
        finally:
            for p in tmp.glob("*"):
                try: p.unlink()
                except Exception: pass
            try: tmp.rmdir()
            except Exception: pass


def main():
    c = connect()
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else "inventory"
        if mode == "inventory": remote_inventory(c)
        elif mode == "download": download(c)
        elif mode == "download-fast": download_fast(c)
        elif mode == "download-carrier-fast": download_carrier_fast()
        elif mode == "stream-token": stream_file("D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag/token_layer_features.npz", rel_local("D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag/token_layer_features.npz"))
        else: raise SystemExit("usage: inventory|download")
    finally:
        c.close()


if __name__ == "__main__": main()
