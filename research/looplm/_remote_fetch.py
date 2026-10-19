"""
Remote-side fetcher for Ouro-1.4B model.safetensors via HTTP from home (Tailscale direct link).
Run on cheems. Downloads the 2.87GB weights, then SHA256-verifies against the canonical hash.
Safe to restart: skips download if file already exists and size matches.
"""
import os, sys, subprocess, hashlib, time

URL = "http://100.66.57.115:8765/model.safetensors"
DST = r"D:\training\Ouro-1.4B\model.safetensors"
EXPECT = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
EXPECT_SIZE = 2869336434
LOG = r"D:\training\fetch_weights.log"

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
    need = True
    if os.path.exists(DST):
        sz = os.path.getsize(DST)
        if sz == EXPECT_SIZE:
            log("File exists with correct size %d. Verifying SHA256..." % sz)
            h = sha256_file(DST)
            if h == EXPECT:
                log("ALREADY COMPLETE: SHA256 matches canonical hash. Nothing to do.")
                return 0
            log("Size OK but hash mismatch (got %s). Re-downloading." % h[:16])
        else:
            log("File exists but size %d != %d. Re-downloading." % (sz, EXPECT_SIZE))
    if need:
        log("Starting curl download: %s -> %s" % (URL, DST))
        t0 = time.time()
        # curl with progress meter to stderr; --retry for resilience
        r = subprocess.run(
            ["curl", "-L", "--retry", "5", "--retry-delay", "5",
             "--connect-timeout", "20", "--max-time", "3600",
             "-o", DST, URL],
            capture_output=True, text=True, timeout=3700)
        el = time.time() - t0
        log("curl exit=%d in %.0fs" % (r.returncode, el))
        if r.stderr:
            for line in r.stderr.splitlines()[-10:]:
                log("  curl: " + line)
        if r.returncode != 0:
            log("CURL FAILED rc=%d" % r.returncode)
            return 1
    sz = os.path.getsize(DST)
    log("Downloaded size: %d (expect %d) %s" % (sz, EXPECT_SIZE, "OK" if sz == EXPECT_SIZE else "MISMATCH"))
    if sz != EXPECT_SIZE:
        log("SIZE MISMATCH, aborting hash check.")
        return 2
    log("Computing SHA256...")
    h = sha256_file(DST)
    log("SHA256: %s" % h)
    log("EXPECT : %s" % EXPECT)
    if h == EXPECT:
        log("VERIFIED: weights match canonical hash.")
        return 0
    log("HASH MISMATCH. Transfer may be corrupt.")
    return 3

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rc = main()
    log("fetch_weights exit code %d" % rc)
    sys.exit(rc)
