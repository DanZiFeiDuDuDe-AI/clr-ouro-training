"""
Fetch Ouro-1.4B model.safetensors from hf-mirror on cheems.
Resumable (curl -C -). SHA256 verified against canonical hash.
Run detached; progress in D:\\training\\fetch_weights_hf.log
"""
import os, sys, subprocess, hashlib, time

URL = "https://hf-mirror.com/ByteDance/Ouro-1.4B/resolve/main/model.safetensors"
DST = r"D:\training\Ouro-1.4B\model.safetensors"
EXPECT = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
EXPECT_SIZE = 2869336434
LOG = r"D:\training\fetch_weights_hf.log"

def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    # if a partial/old file exists with wrong size, curl -C - will resume; if hash already good, skip
    if os.path.exists(DST):
        sz = os.path.getsize(DST)
        if sz == EXPECT_SIZE:
            log("File complete (%d B). Verifying SHA256...", )
            h = sha256_file(DST)
            if h == EXPECT:
                log("VERIFIED (already complete). SHA256 matches canonical.")
                return 0
            log("Size OK but hash %s != expect. Deleting and re-downloading.", h[:16])
            os.remove(DST)
        elif sz > EXPECT_SIZE:
            log("File larger than expected (%d>%d). Deleting.", sz, EXPECT_SIZE)
            os.remove(DST)
        else:
            log("Partial file %d/%d B. Resuming.", sz, EXPECT_SIZE)

    log("Starting resumable curl download from hf-mirror...")
    t0 = time.time()
    r = subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "10", "--retry-delay", "5",
         "--connect-timeout", "20", "--max-time", "3600",
         "-o", DST, URL],
        capture_output=True, text=True, timeout=3700)
    el = time.time() - t0
    log("curl exit=%d in %.0fs" % (r.returncode, el))
    if r.stderr:
        for line in r.stderr.splitlines()[-12:]:
            log("  curl: " + line)
    if r.returncode != 0:
        log("CURL FAILED rc=%d (may be partial; re-run to resume)" % r.returncode)
        return 1

    sz = os.path.getsize(DST)
    log("Downloaded size: %d (expect %d) %s" % (sz, EXPECT_SIZE, "OK" if sz == EXPECT_SIZE else "MISMATCH"))
    if sz != EXPECT_SIZE:
        log("SIZE MISMATCH. Re-run to resume.")
        return 2

    log("Computing SHA256 (this takes ~10s)...")
    h = sha256_file(DST)
    log("SHA256: %s" % h)
    log("EXPECT : %s" % EXPECT)
    if h == EXPECT:
        log("VERIFIED: weights match canonical hash. Transfer complete.")
        return 0
    log("HASH MISMATCH. File may be corrupt. Deleting for clean re-download.")
    os.remove(DST)
    return 3

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rc = main()
    log("fetch_hf exit code %d" % rc)
    sys.exit(rc)
