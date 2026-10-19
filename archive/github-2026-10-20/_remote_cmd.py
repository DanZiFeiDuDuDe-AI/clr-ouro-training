"""Run a command on cheems via paramiko and print stdout/stderr.

Usage: python _remote_cmd.py <command...>
If no command given, runs a default status probe.
"""
import sys, warnings
import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "cheems"
USER = "ai training"
PASSWORD = "REPLACE_ME"

def run(cmd, timeout=120):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=20, banner_timeout=30, auth_timeout=30)
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err
    finally:
        c.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = "hostname & whoami & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
    rc, out, err = run(cmd)
    print("=== CMD ===")
    print(cmd)
    print("=== RC ===")
    print(rc)
    print("=== STDOUT ===")
    print(out)
    print("=== STDERR ===")
    print(err)
