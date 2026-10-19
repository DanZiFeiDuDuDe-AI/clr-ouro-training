"""Test cheems public-internet reachability to Ouro download sources."""
import urllib.request, ssl, time, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

urls = [
    "https://huggingface.co",
    "https://hf-mirror.com",
    "https://www.modelscope.cn",
    "https://modelscope.cn",
    # Direct file probe on hf-mirror (ByteDance/Ouro-1.4B)
    "https://hf-mirror.com/ByteDance/Ouro-1.4B/resolve/main/config.json",
    "https://huggingface.co/ByteDance/Ouro-1.4B/resolve/main/config.json",
]
for u in urls:
    try:
        req = urllib.request.Request(u, method="GET", headers={"User-Agent": "curl/8.0"})
        t = time.time()
        r = urllib.request.urlopen(req, timeout=15, context=ctx)
        dt = time.time() - t
        body = r.read(200)
        print("OK  %-65s %3d  %.2fs  %r" % (u, r.status, dt, body[:40]))
    except Exception as e:
        print("FAIL %-65s  %s" % (u, str(e)[:90]))
