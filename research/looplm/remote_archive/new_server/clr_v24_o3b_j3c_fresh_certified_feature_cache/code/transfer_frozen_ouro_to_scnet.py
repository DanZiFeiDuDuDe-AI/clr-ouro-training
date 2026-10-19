"""Copy the pinned frozen Ouro artifact from Cheems to the active scnet host.

Credentials are never persisted or printed; the destination password is read
with getpass. The source connection reuses the existing local helper.
"""
from __future__ import annotations

import getpass
import sys
import hashlib
import socket
import time
from pathlib import Path

import paramiko

HOST = "ssh.zzai.scnet.cn"
PORT = 11810
USER = "root"
LOCAL_ASSETS = Path(__file__).parents[1] / "source_snapshot" / "Ouro-1.4B"
REMOTE_ROOT = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache/source_snapshot/Ouro-1.4B"
SOURCE_WEIGHT = "D:/training/Ouro-1.4B/model.safetensors"
TARGET_WEIGHT = REMOTE_ROOT + "/model.safetensors"
EXPECTED_WEIGHT_BYTES = 2_869_336_434
EXPECTED_WEIGHT_SHA256 = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
ASSETS = (
    "config.json",
    "configuration_ouro.py",
    "modeling_ouro.py",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class LargeWindowTransport(paramiko.Transport):
    def __init__(self, sock, *args, **kwargs):
        kwargs["default_window_size"] = 128 * 1024 * 1024
        kwargs["default_max_packet_size"] = 4 * 1024 * 1024
        super().__init__(sock, *args, **kwargs)


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def sha256_local(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    password = getpass.getpass(f"SSH password for {USER}@{HOST}:{PORT}: ")
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target.connect(HOST, port=PORT, username=USER, password=password, timeout=30, banner_timeout=30, auth_timeout=30, transport_factory=LargeWindowTransport)
    del password
    target_sftp = target.open_sftp()
    try:
        mkdirs(target_sftp, REMOTE_ROOT)
        for name in ASSETS:
            local = LOCAL_ASSETS / name
            remote = REMOTE_ROOT + "/" + name
            target_sftp.put(str(local), remote)
            if target_sftp.stat(remote).st_size != local.stat().st_size:
                raise IOError(f"asset size mismatch: {name}")
            print(f"ASSET_OK {name} bytes={local.stat().st_size}", flush=True)

        local_weight = LOCAL_ASSETS / "model.safetensors"
        if local_weight.stat().st_size != EXPECTED_WEIGHT_BYTES:
            raise IOError(f"local pinned weight size mismatch: {local_weight.stat().st_size}")
        local_hash = sha256_local(local_weight)
        if local_hash != EXPECTED_WEIGHT_SHA256:
            raise IOError(f"local pinned weight SHA mismatch: {local_hash}")
        try:
            target_stat = target_sftp.stat(TARGET_WEIGHT)
            if target_stat.st_size == EXPECTED_WEIGHT_BYTES:
                print("WEIGHT_ALREADY_PRESENT size matches; leaving it untouched", flush=True)
                return
        except OSError:
            pass

        print(f"WEIGHT_TRANSFER_BEGIN bytes={local_weight.stat().st_size} sha256={local_hash}", flush=True)
        started = time.monotonic()
        transferred = 0
        last_log = 0
        with local_weight.open("rb") as src, target_sftp.open(TARGET_WEIGHT, "wb") as dst:
            dst.set_pipelined(True)
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                transferred += len(block)
                if transferred - last_log >= 256 * 1024 * 1024:
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(f"WEIGHT_PROGRESS bytes={transferred} rate_MiB_s={transferred/elapsed/1024**2:.2f}", flush=True)
                    last_log = transferred
            dst.flush()
        actual = target_sftp.stat(TARGET_WEIGHT).st_size
        if transferred != EXPECTED_WEIGHT_BYTES or actual != EXPECTED_WEIGHT_BYTES:
            raise IOError(f"weight transfer size mismatch: sent={transferred}, remote={actual}")
        _, stdout, stderr = target.exec_command(f"sha256sum '{TARGET_WEIGHT}'", timeout=600)
        remote_hash = stdout.read().decode("utf-8", "replace").split()[0]
        err = stderr.read().decode("utf-8", "replace")
        if remote_hash != EXPECTED_WEIGHT_SHA256:
            raise IOError(f"remote pinned weight SHA mismatch: {remote_hash}; {err[-500:]}")
        print(f"WEIGHT_TRANSFER_COMPLETE bytes={actual} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        print(f"WEIGHT_SHA256_VERIFIED {remote_hash}", flush=True)
    finally:
        target_sftp.close()
        target.close()


if __name__ == "__main__":
    main()
