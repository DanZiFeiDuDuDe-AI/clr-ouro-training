"""Remote launcher for verify_baseline.py on cheems (Phase 0B reproduction)."""
import os, sys, runpy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_ROOT"] = "D:/training"
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
os.environ["OURO_CKPT"] = "D:/training/checkpoints/clr_ouro_v1_s42/best_model.pt"
sys.path.insert(0, "D:/training")
sys.path.insert(0, "D:/training/exp_seqstate_s0")

runpy.run_path("D:/training/exp_seqstate_s0/verify_baseline.py",
               run_name="__main__")
