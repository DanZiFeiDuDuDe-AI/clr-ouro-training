from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from pathlib import Path

import paramiko


OUT = Path(__file__).resolve().parent
REMOTE_QUERIES = [
    (
        "exact_component_names",
        "timeout 60s find /root/private_data -maxdepth 8 \\( -type d \\( -name cache -o -name source_snapshot -o -name token_layer_features \\) -prune \\) -o -type f \\( -iname 'a1_v24_o21a.py' -o -iname 'a1_v24_o21a_mr.py' -o -iname 'a1_v24_vgen.py' -o -iname 'a1_v24_o3a.py' -o -iname '*o21a*best*.pt' -o -iname '*vgen2*' -o -iname 'allocator_kaware_best.pt' -o -iname 'router_best.pt' -o -iname '*o3b_update*' \\) -printf '%p\\t%s\\t%TY-%Tm-%TdT%TH:%TM:%TS\\n' 2>/dev/null",
    ),
    (
        "clr_related_checkpoint_files",
        "timeout 60s find /root/private_data/clr_v24_o3b_l/checkpoints -maxdepth 5 -type f \\( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.safetensors' \\) -printf '%p\\t%s\\n' 2>/dev/null | grep -E 'clr_v24_(o21|o3a|o3b|vgen)'",
    ),
    (
        "configs_code_and_logs_references",
        "timeout 30s grep -RIlE --binary-files=without-match --include='*.py' --include='*.json' --include='*.yaml' --include='*.yml' --include='*.log' --include='*.txt' '(o21a|vgen2|allocator_kaware|router_best|o3b_update)' /root/private_data/clr_v24_o3b_l/output /root/private_data/clr_v24_o3b_l/context /public/home/ac92f4k0z9/clr_v24_o3b_l/output /public/home/ac92f4k0z9/clr_v24_o3b_l/context 2>/dev/null",
    ),
    (
        "target_experiment_directories",
        "timeout 60s find /root/private_data -maxdepth 6 -type d \\( -iname '*o21a*' -o -iname '*vgen2*' -o -iname '*o3a*' -o -iname '*o3b*update*' \\) -printf '%p\\n' 2>/dev/null",
    ),
    (
        "other_storage_roots_exact_checkpoints",
        "timeout 20s find /public/home/ac92f4k0z9 /public/share/ac92f4k0z9 /public/SothisAI/autosave -maxdepth 8 \\( -type d \\( -name cache -o -name .cache -o -name source_snapshot \\) -prune \\) -o -type f \\( -iname 'best_model.pt' -o -iname 'best.pt' -o -iname 'allocator_kaware_best.pt' -o -iname 'router_best.pt' -o -iname '*o21a*.pt' -o -iname '*vgen2*.pt' \\) -printf '%p\\t%s\\n' 2>/dev/null",
    ),
]


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
    results = {"host": "ssh.zzai.scnet.cn:11810", "searched_utc": datetime.now(timezone.utc).isoformat(), "queries": {}}
    try:
        sftp = client.open_sftp()
        for path in ("/", "/root", "/root/private_data", "/root/public_data", "/root/group_data", "/public", "/home", "/tmp", "/mnt", "/workspace", "/root/archive"):
            try:
                entries = sorted(item.filename for item in sftp.listdir_attr(path))
                results.setdefault("directory_inventory", {})[path] = entries[:250]
            except OSError as exc:
                results.setdefault("directory_inventory", {})[path] = {"not_available": type(exc).__name__}
        for path in (
            "/root/private_data/clr_v24_o3b_l",
            "/root/private_data/clr_v24_o3b_l/code",
            "/root/private_data/clr_v24_o3b_l/checkpoints",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_vgen2",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3a_allocation",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_update",
            "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_m_operational_samekey_rescue",
            "/root/private_data/clr_v24_o3b_l/output",
            "/root/private_data/clr_v24_o3b_l/context",
            "/public/home/ac92f4k0z9/clr_v24_o3b_l",
            "/public/home/ac92f4k0z9/clr_v24_o3b_l/output",
            "/public/home/ac92f4k0z9/clr_v24_o3b_l/checkpoints",
            "/root/public_data/data",
            "/root/public_data/other",
            "/root/public_data/xiaodao2026",
            "/public/SothisAI",
            "/public/home",
            "/public/share",
            "/public/home/ac92f4k0z9",
            "/public/share/ac92f4k0z9",
            "/public/SothisAI/autosave",
        ):
            try:
                entries = sorted((item.filename, item.st_size, item.st_mtime) for item in sftp.listdir_attr(path))
                results.setdefault("target_path_inventory", {})[path] = entries[:300]
            except OSError as exc:
                results.setdefault("target_path_inventory", {})[path] = {"not_available": type(exc).__name__}
        source_files = (
            "a1_v24_o21a.py",
            "a1_v24_o21a_mr.py",
            "a1_v24_vgen.py",
            "a1_v24_o3a.py",
            "a1_v24_o3b.py",
            "a1_v24_o2_common.py",
        )
        results["original_source_file_hashes"] = {}
        for name in source_files:
            remote_path = "/root/private_data/clr_v24_o3b_l/code/" + name
            try:
                digest = __import__("hashlib").sha256()
                with sftp.open(remote_path, "rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                attr = sftp.stat(remote_path)
                results["original_source_file_hashes"][remote_path] = {"sha256": digest.hexdigest(), "bytes": attr.st_size}
            except OSError as exc:
                results["original_source_file_hashes"][remote_path] = {"not_available": type(exc).__name__}
        sftp.close()
        (OUT / "remote_dependency_search.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, command in REMOTE_QUERIES:
            stdin, stdout, stderr = client.exec_command(command, timeout=110)
            stdin.close()
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            results["queries"][name] = {
                "exit_code": stdout.channel.recv_exit_status(),
                "stdout": out,
                "stderr": err[-4000:],
            }
            (OUT / "remote_dependency_search.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    finally:
        client.close()
    path = OUT / "remote_dependency_search.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"REMOTE_SEARCH_SAVED={path}")
    for name, result in results["queries"].items():
        print(f"--- {name} (exit={result['exit_code']}) ---")
        print(result["stdout"][:15000] or "<no matches>")
        if result["stderr"].strip():
            print("STDERR_TAIL=" + result["stderr"][-1000:])
    print("--- remote locations ---")
    print(json.dumps(results.get("directory_inventory", {}), ensure_ascii=False))
    print("--- target path inventory ---")
    print(json.dumps(results.get("target_path_inventory", {}), ensure_ascii=False))
    print("--- original source file hashes ---")
    print(json.dumps(results.get("original_source_file_hashes", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
