from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath

import paramiko


LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m2_growing_slot_calibration"
REFRESHABLE = {
    "final_report.md",
    "FINAL_VERDICT.txt",
    "final_verdict.json",
    "artifact_sha256_manifest.json",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_remote(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    shell_command = (
        "bash -lc 'export LD_LIBRARY_PATH=/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:"
        f"/opt/dtk-26.04/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH; {command}'"
    )
    stdin, stdout, stderr = client.exec_command(shell_command, timeout=180)
    stdin.close()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "ssh.zzai.scnet.cn",
        port=11810,
        username="root",
        password=getpass.getpass("SSH password: "),
        timeout=40,
        auth_timeout=40,
        banner_timeout=40,
        allow_agent=False,
        look_for_keys=False,
    )
    sftp = client.open_sftp()
    try:
        remote_script = REMOTE + "/code/o3b_m2_runner_v6.py"
        sftp.put(str(LOCAL / "o3b_m2_runner.py"), remote_script)
        status, out, err = run_remote(client, f"python3 {remote_script} --phase finalize")
        log = f"STDOUT:\n{out}\nSTDERR:\n{err}\nEXIT={status}\n"
        with sftp.open(REMOTE + "/logs/finalize_v6.log", "w") as stream:
            stream.write(log)
        if status:
            print(log)
            return status

        status, out, err = run_remote(client, f"python3 {remote_script} --phase finalize")
        if status:
            print(f"STDOUT:\n{out}\nSTDERR:\n{err}\nEXIT={status}")
            return status
        print("REMOTE_FINALIZE=" + out.strip())
        if err.strip():
            print("REMOTE_STDERR=" + err.strip())

        remote_manifest_path = REMOTE + "/artifact_sha256_manifest.json"
        manifest = json.loads(sftp.open(remote_manifest_path, "r").read().decode("utf-8"))
        for record in manifest["files"]:
            rel = PurePosixPath(record["path"])
            remote_path = REMOTE + "/" + rel.as_posix()
            local_path = LOCAL.joinpath(*rel.parts)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            remote_digest = sha_remote(sftp, remote_path)
            if remote_digest != record["sha256"]:
                raise IOError(f"Remote manifest mismatch: {rel}")
            if local_path.exists() and sha_file(local_path) == remote_digest:
                continue
            is_root_refresh = len(rel.parts) == 1 and rel.name in REFRESHABLE
            if local_path.exists() and not is_root_refresh:
                raise FileExistsError(f"Refusing to replace changed artifact: {rel}")
            staging = local_path.with_name(local_path.name + ".verified-refresh")
            sftp.get(remote_path, str(staging))
            if sha_file(staging) != remote_digest:
                staging.unlink(missing_ok=True)
                raise IOError(f"Downloaded SHA-256 mismatch: {rel}")
            staging.replace(local_path)

        local_manifest = LOCAL / "artifact_sha256_manifest.json"
        expected_manifest_sha = sha_remote(sftp, remote_manifest_path)
        staging = local_manifest.with_name(local_manifest.name + ".verified-refresh")
        sftp.get(remote_manifest_path, str(staging))
        if sha_file(staging) != expected_manifest_sha:
            staging.unlink(missing_ok=True)
            raise IOError("Downloaded manifest SHA-256 mismatch")
        staging.replace(local_manifest)
        print(f"VERIFIED_ARTIFACTS={len(manifest['files'])}")
        print(f"LOCAL_OUTPUT={LOCAL}")
        return 0
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
