from __future__ import annotations

import argparse
import getpass
import sys

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="remote command to run")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            "ssh.zzai.scnet.cn",
            port=11810,
            username="root",
            password=getpass.getpass("SSH password: "),
            timeout=30,
            auth_timeout=30,
            banner_timeout=30,
        )
        stdin, stdout, stderr = client.exec_command(args.command, timeout=args.timeout)
        stdin.close()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
