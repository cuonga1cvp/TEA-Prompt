"""
Visualize attention drift between task 0 (old) and task 1 (current) under 3
loss configurations.

Design:
  - Phase A trains task 0 once with no extra losses, saving a SHARED
    snapshot_task0.pt under outputs/compare_losses/_shared/snapshots/.
  - Each Phase B case loads that shared task-0 state, trains task 1 with
    its own loss config, and saves snapshot_task1.pt under its case dir.

This script:
  1. Loads the shared (pk_0, pv_0, query_0) once → ONE attention_0 anchor.
  2. For each case, loads (pk_1, pv_1) and computes attention_0*.
  3. Fits ONE PCA on the union {attention_0, attn_0*_case for each case} so
     all 3 plots share the same coordinate system AND identical anchor points.
  4. Saves 3 PNG files, one per case.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUT_BASE = "outputs/compare_losses"
SHARED_SNAP = os.path.join(OUT_BASE, "_shared", "snapshots", "snapshot_task0.pt")

CASES = [
    ("no_loss",    "No proto, no attention"),
    ("proto_only", "Only proto loss"),
    ("attn_only",  "Only attention loss"),
]


def compute_attention(pk_dict, pv_dict, query_dict):
    """One attention output vector per (layer, head). Returns (num_pairs, D)."""
    rows = []
    for key in sorted(pk_dict.keys()):
        K = pk_dict[key]   # (E, D)
        V = pv_dict[key]   # (E, D)
        q = query_dict[key]  # (1, D)
        scores = q @ K.t()
        weights = torch.softmax(scores, dim=-1)
        attn = (weights @ V).squeeze(0)
        rows.append(attn.numpy())
    return np.stack(rows, axis=0)


def main():
    # 1. Load shared anchor (pk_0, pv_0, query_0)
    if not os.path.exists(SHARED_SNAP):
        print(f"[fatal] shared snapshot not found: {SHARED_SNAP}")
        print("Run experiments/compare_losses.sh first to generate it.")
        return
    shared = torch.load(SHARED_SNAP, map_location="cpu")
    pk_0, pv_0, query_0 = shared["pk"], shared["pv"], shared["query"]
    attn_0 = compute_attention(pk_0, pv_0, query_0)        # (72, D)

    # 2. Per-case attn_0*
    attn_stars = {}
    for case, _ in CASES:
        snap1 = os.path.join(OUT_BASE, case, "snapshots", "snapshot_task1.pt")
        if not os.path.exists(snap1):
            print(f"[skip] {case}: missing {snap1}")
            continue
        s1 = torch.load(snap1, map_location="cpu")
        attn_stars[case] = compute_attention(s1["pk"], s1["pv"], query_0)

    if not attn_stars:
        print("No cases found, nothing to plot.")
        return

    # 3. Fit ONE PCA on union — locks anchor positions across all plots
    union = np.concatenate(
        [attn_0] + [v for v in attn_stars.values()],
        axis=0,
    )
    pca = PCA(n_components=2)
    pca.fit(union)
    proj_anchor = pca.transform(attn_0)
    proj_stars = {c: pca.transform(v) for c, v in attn_stars.items()}

    # Compute global axis limits so all 3 plots share the same view
    all_proj = np.concatenate([proj_anchor] + list(proj_stars.values()), axis=0)
    pad_x = 0.05 * (all_proj[:, 0].max() - all_proj[:, 0].min() + 1e-9)
    pad_y = 0.05 * (all_proj[:, 1].max() - all_proj[:, 1].min() + 1e-9)
    xlim = (all_proj[:, 0].min() - pad_x, all_proj[:, 0].max() + pad_x)
    ylim = (all_proj[:, 1].min() - pad_y, all_proj[:, 1].max() + pad_y)

    # 4. Plot per case
    for case, title in CASES:
        if case not in attn_stars:
            continue
        p1 = proj_anchor
        p2 = proj_stars[case]
        attn_0_star = attn_stars[case]

        # Drift metrics in original D-dim space
        l2_drift = np.linalg.norm(attn_0 - attn_0_star, axis=1).mean()
        cos = (attn_0 * attn_0_star).sum(axis=1) / (
            np.linalg.norm(attn_0, axis=1)
            * np.linalg.norm(attn_0_star, axis=1)
            + 1e-12
        )
        cossim = cos.mean()

        fig, ax = plt.subplots(figsize=(8, 6.5))
        for i in range(p1.shape[0]):
            ax.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]],
                    color="gray", alpha=0.35, linewidth=0.6)
        ax.scatter(p1[:, 0], p1[:, 1], c="tab:blue", s=45, alpha=0.85,
                   label=r"$\mathrm{attention}_0$  (shared anchor, task 0)")
        ax.scatter(p2[:, 0], p2[:, 1], c="tab:red", s=45, alpha=0.85,
                   marker="x",
                   label=r"$\mathrm{attention}_0^*$  (after task 1)")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(
            f"{title}\nmean L2 drift = {l2_drift:.4f}   |   "
            f"mean cossim = {cossim:.4f}   |   pairs = {p1.shape[0]}"
        )
        ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(OUT_BASE, f"attention_drift_{case}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}   (drift={l2_drift:.4f}, cossim={cossim:.4f})")


if __name__ == "__main__":
    main()
