"""Read-only health check for the active J3C freeze process."""
import getpass
import paramiko

HOST, PORT, USER = "ssh.zzai.scnet.cn", 11810, "root"
CMD = "ps -eo pid,etime,pcpu,pmem,args | grep '[j]3c_extract.py extract'; tail -n 10 /root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache/logs/feature_extraction.log 2>/dev/null; tail -n 8 /root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache/logs/feature_extraction.stderr.log 2>/dev/null; find /root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache -maxdepth 1 -name 'H_fresh_*.npy' -printf '%f %s\n' | tail -10"


def main():
    pw = getpass.getpass(f"SSH password for {USER}@{HOST}:{PORT}: ")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=30)
    del pw
    try:
        _, out, err = c.exec_command(CMD, timeout=30)
        print(out.read().decode("utf-8", "replace"))
        error = err.read().decode("utf-8", "replace")
        if error:
            print("STDERR", error)
    finally:
        c.close()


if __name__ == "__main__":
    main()
