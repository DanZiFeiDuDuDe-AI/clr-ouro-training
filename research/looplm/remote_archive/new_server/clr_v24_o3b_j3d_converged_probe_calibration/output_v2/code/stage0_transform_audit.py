from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache")
OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration_v2")
SPLITS = ("train", "val")
LIMITS = {"max_abs_error": 5e-4, "relative_fro_error": 2e-6,
          "min_row_cosine": 0.9999999, "min_pair_distance_correlation": 0.999999,
          "max_orthogonality_error": 1e-5, "max_weight_transfer_logit_error": 1e-9}


def compare(actual: np.ndarray, expected: np.ndarray, rng: np.random.Generator) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    d = a - e
    rel = np.linalg.norm(d) / max(np.linalg.norm(e), 1e-30)
    an = np.linalg.norm(a, axis=1)
    en = np.linalg.norm(e, axis=1)
    cos = np.sum(a * e, axis=1) / np.maximum(an * en, 1e-30)
    n = len(a)
    left = rng.integers(0, n, size=min(20000, max(1000, n * 2)))
    right = rng.integers(0, n, size=len(left))
    da = np.linalg.norm(a[left] - a[right], axis=1)
    de = np.linalg.norm(e[left] - e[right], axis=1)
    corr = float(np.corrcoef(da, de)[0, 1]) if np.std(da) and np.std(de) else None
    return {"shape": list(a.shape), "max_abs_error": float(np.max(np.abs(d))),
            "mean_abs_error": float(np.mean(np.abs(d))), "relative_fro_error": float(rel),
            "row_cosine_min": float(np.min(cos)), "row_cosine_mean": float(np.mean(cos)),
            "pair_distance_correlation": corr}


def classifier_transfer_check(rng: np.random.Generator, A: np.ndarray, direction: str) -> dict:
    n, d, classes = 96, A.shape[0], 7
    xs = rng.normal(size=(n, d))
    xd = xs @ A
    ws = rng.normal(size=(classes, d))
    bs = rng.normal(size=classes)
    wd = ws @ A
    raw_error = float(np.max(np.abs((xs @ ws.T + bs) - (xd @ wd.T + bs))))

    mu_s, sd_s = xs.mean(0), xs.std(0)
    mu_d, sd_d = xd.mean(0), xd.std(0)
    sd_s[sd_s < 1e-8] = 1.0
    sd_d[sd_d < 1e-8] = 1.0
    B = (sd_s[:, None] * A) / sd_d[None, :]
    offset = (mu_s @ A - mu_d) / sd_d
    wd_std = np.linalg.solve(B, ws.T).T
    bd_std = bs - offset @ wd_std.T
    xss = (xs - mu_s) / sd_s
    xds = (xd - mu_d) / sd_d
    standardized_error = float(np.max(np.abs((xss @ ws.T + bs) - (xds @ wd_std.T + bd_std))))
    return {"direction": direction, "classes": classes,
            "raw_max_logit_error": raw_error,
            "standardized_max_logit_error": standardized_error}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity").mkdir(parents=True, exist_ok=True)
    with np.load(CACHE / "w1_svd_matrices.npz", allow_pickle=False) as data:
        keys = list(data.files)
        U = np.asarray(data["U"] if "U" in keys else data["u"], dtype=np.float64)
        s_key = next((key for key in ("S", "s", "singular_values") if key in keys), None)
        if s_key is None:
            raise KeyError(f"singular values missing; keys={keys}")
        S = np.asarray(data[s_key], dtype=np.float64).reshape(-1)
        vh_key = next((key for key in ("Vh", "vh", "Vt") if key in keys), None)
        Vh_shape = list(data[vh_key].shape) if vh_key else None
        dtypes = {key: str(data[key].dtype) for key in keys}

    orth_err = float(np.max(np.abs(U.T @ U - np.eye(U.shape[1]))))
    rng = np.random.default_rng(20260925)
    results = {"experiment": "O3b-J3D Stage 0 row-transform/equivalence audit",
               "cache_path": str(CACHE), "svd_keys": keys, "svd_dtypes": dtypes,
               "U_shape": list(U.shape), "S_shape": list(S.shape), "Vh_shape": Vh_shape,
               "U_orthogonality_max_abs_error": orth_err,
               "singular_values_min": float(np.min(S)), "singular_values_max": float(np.max(S)),
               "definitions": {"R0": "V^T H, cached orthonormal row coordinates",
                               "R1": "R0 * singular_values, same coordinates with Sigma weighting",
                               "R2": "R1 @ U.T, direct W1H up to the stored factorization"},
               "split_checks": {}, "weight_transfer_synthetic_checks": [], "limits": LIMITS}
    all_pass = (U.shape == (256, 256) and S.shape == (256,) and Vh_shape == [256, 2048]
                and orth_err <= LIMITS["max_orthogonality_error"] and np.all(S > 0))
    for split in SPLITS:
        R0 = np.asarray(np.load(CACHE / f"R0_{split}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
        R1 = np.asarray(np.load(CACHE / f"R1_{split}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
        R2 = np.asarray(np.load(CACHE / f"R2_{split}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
        r0_r1 = compare(R1, R0 * S[None, :], rng)
        r1_r2 = compare(R2, R1 @ U.T, rng)
        checks = {"R0_to_R1_sigma_weighting": r0_r1, "R1_to_R2_orthogonal_rotation": r1_r2}
        checks["pass"] = all(
            item["max_abs_error"] <= LIMITS["max_abs_error"]
            and item["relative_fro_error"] <= LIMITS["relative_fro_error"]
            and item["row_cosine_min"] >= LIMITS["min_row_cosine"]
            and item["pair_distance_correlation"] >= LIMITS["min_pair_distance_correlation"]
            for item in (r0_r1, r1_r2)
        )
        results["split_checks"][split] = checks
        all_pass &= checks["pass"]

    d = U.shape[0]
    for A, name in ((U.T, "R1_to_R2"), (U, "R2_to_R1")):
        check = classifier_transfer_check(rng, A, name)
        results["weight_transfer_synthetic_checks"].append(check)
        all_pass &= check["raw_max_logit_error"] <= LIMITS["max_weight_transfer_logit_error"]
        all_pass &= check["standardized_max_logit_error"] <= LIMITS["max_weight_transfer_logit_error"]

    results["pass"] = bool(all_pass)
    results["test_metrics_or_selection_used"] = False
    results["checkpoint_modified"] = False
    results["elapsed_seconds"] = time.time() - t0
    (OUT / "sanity" / "stage0_transform_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "stage0_audit.md").write_text(
        "# O3b-J3D Stage 0 — Transform and equivalence audit\n\n"
        f"Verdict: **{'PASS' if all_pass else 'FAIL'}**. No probes were trained; no held-out metrics or model weights were read/changed.\n\n"
        f"SVD shapes: U={U.shape}, Sigma={S.shape}, Vh={Vh_shape}; U orthogonality max error={orth_err:.3e}.\n\n"
        "For train and validation, cached R1 was checked against R0·Sigma and cached R2 against R1·Uᵀ, including maximum/mean error, relative Frobenius error, per-row cosine, and sampled pair-distance correlation. A synthetic 7-class affine classifier transfer was checked in both directions on raw and train-standardized coordinates.\n\n"
        + "\n".join(f"- {split}: {'PASS' if values['pass'] else 'FAIL'}; R0→R1 max abs {values['R0_to_R1_sigma_weighting']['max_abs_error']:.3e}; R1→R2 max abs {values['R1_to_R2_orthogonal_rotation']['max_abs_error']:.3e}" for split, values in results["split_checks"].items())
        + "\n\nClassifier transfer synthetic checks:\n\n"
        + "\n".join(f"- {x['direction']}: raw max logit error={x['raw_max_logit_error']:.3e}, standardized={x['standardized_max_logit_error']:.3e}" for x in results["weight_transfer_synthetic_checks"])
        + "\n", encoding="utf-8")
    print(json.dumps({"stage": "stage0_transform_audit_complete", "pass": bool(all_pass),
                      "elapsed_s": results["elapsed_seconds"], "svd_keys": keys}, ensure_ascii=False), flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
