"""Authenticated remote execution and atomic SHA-256 verified checkpoint pulls.

Credentials are read from the current Windows user's DPAPI-protected file.
No passwords are written to logs, commands, manifests, or experiment configs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import ctypes
from ctypes import wintypes
import time
import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT, USER = 'ssh.zzai.scnet.cn', 11810, 'root'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def connect(profile='scnet'):
    if profile=='cheems':
        # Read the old, already-local connection definition without executing it.
        import ast
        values={}
        for node in ast.parse((ROOT/'_remote_cmd.py').read_text(encoding='utf-8-sig')).body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=node.value.value
        client=paramiko.SSHClient();client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(values['HOST'],username=values['USER'],password=values['PASSWORD'],look_for_keys=False,allow_agent=False,timeout=12,auth_timeout=12,banner_timeout=12)
            return client
        except BaseException:
            client.close();raise
    if profile!='scnet':raise ValueError('Unknown project SSH profile')
    password = os.environ.get('LOOPLM_SSH_PASSWORD')
    if not password:
        credential = ROOT / '.local' / 'scnet_password.dpapi'
        class Blob(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
        raw = bytes.fromhex(credential.read_text(encoding='ascii').strip())
        buf = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        source, output = Blob(len(raw), buf), Blob()
        api = ctypes.windll.crypt32.CryptUnprotectData
        if not api(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
            raise RuntimeError('Cannot decrypt local SSH credential for this Windows user')
        try: password = ctypes.string_at(output.data, output.size).decode('utf-16-le')
        finally: ctypes.windll.kernel32.LocalFree(output.data)
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        c.connect(HOST, port=PORT, username=USER, password=password,
                  look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
        c.get_transport().set_keepalive(20)
        return c
    except BaseException:
        c.close()
        raise

def execute(command, timeout=60):
    c = connect()
    try:
        stdin, out, err = c.exec_command(command, timeout=timeout)
        stdin.close()
        text, error = out.read().decode('utf-8', 'replace'), err.read().decode('utf-8', 'replace')
        code = out.channel.recv_exit_status()
        if code: raise RuntimeError(f'Remote command failed ({code}): {error[-2000:]}')
        return text
    finally: c.close()

def pull(remote, local, extensions=None, exclude_names=()):
    """Reconnect on every call; never follow symlinks; do not overwrite differing files."""
    local = Path(local).resolve()
    local.mkdir(parents=True, exist_ok=True)
    remote = str(PurePosixPath(remote))
    c = connect()
    sftp = c.open_sftp()
    records = []
    try:
        def walk(folder, rel=PurePosixPath('.')):
            for item in sorted(sftp.listdir_attr(folder), key=lambda x: x.filename):
                if item.filename in ('.', '..') or '/' in item.filename or '\\' in item.filename: raise ValueError('Unsafe remote filename')
                rp = folder.rstrip('/') + '/' + item.filename
                sub = rel / item.filename
                if stat.S_ISDIR(item.st_mode): walk(rp, sub)
                elif stat.S_ISREG(item.st_mode) and item.filename not in exclude_names and (extensions is None or PurePosixPath(item.filename).suffix in extensions):
                    target = (local / Path(*sub.parts)).resolve()
                    if not target.is_relative_to(local): raise ValueError('Unsafe target path')
                    # Hash on the server avoids a second full transfer of each weight.
                    _, out, err = c.exec_command('sha256sum -- ' + shlex.quote(rp), timeout=120)
                    result = out.read().decode(); error = err.read().decode()
                    if out.channel.recv_exit_status(): raise RuntimeError(error)
                    expected = result.split()[0]
                    if target.exists():
                        if digest(target) != expected: raise FileExistsError(f'Local artifact differs; preserved: {target}')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temp = target.with_name(target.name + '.partial')
                        before = sftp.stat(rp)
                        sftp.get(rp, str(temp))
                        after = sftp.stat(rp)
                        if before.st_size != after.st_size or before.st_mtime != after.st_mtime or digest(temp) != expected:
                            raise IOError(f'Changed or corrupt artifact: {rp}')
                        temp.replace(target)
                    records.append({'path': str(sub), 'bytes': item.st_size, 'sha256': expected})
        walk(remote)
        manifest = {'host': HOST, 'port': PORT, 'remote': remote, 'verified_files': records,
                    'verified_at_unix': time.time(), 'weights': sum(Path(r['path']).suffix in {'.pt','.pth','.safetensors','.ckpt'} for r in records)}
        manifest_path = local / 'download_verification.json'
        temp = manifest_path.with_suffix('.json.partial')
        temp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        temp.replace(manifest_path)
        return manifest
    finally:
        sftp.close(); c.close()

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='action', required=True)
    cmd = sub.add_parser('exec'); cmd.add_argument('command'); cmd.add_argument('--timeout', type=int, default=60)
    p = sub.add_parser('pull'); p.add_argument('remote'); p.add_argument('local', type=Path); p.add_argument('--weights-only', action='store_true')
    args = ap.parse_args()
    if args.action == 'exec': print(execute(args.command, args.timeout))
    else:
        manifest = pull(args.remote, args.local, {'.pt','.pth','.safetensors','.ckpt'} if args.weights_only else None)
        print(json.dumps({'files_verified': len(manifest['verified_files']), 'weights': manifest['weights'], 'local': str(args.local)}))

if __name__ == '__main__': main()
