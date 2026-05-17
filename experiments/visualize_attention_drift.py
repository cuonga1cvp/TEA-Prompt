"""
Visualize per-expert attention drift between task 0 (old) and task 1 (current)
under 3 loss configurations.

For each (layer e, head h, expert l), the per-expert contribution to the
attention output under the SHARED anchor query q_0 is

    contrib_old[l]     = softmax(q_0 @ pk_old.T)[l]     * pv_old[l]      (D,)
    contrib_current[l] = softmax(q_0 @ pk_current.T)[l] * pv_current[l]  (D,)

Only experts whose ANCHOR routing score `q_0 @ pk_old.T` is in the **top-K**
per (layer, head) are kept (default K=5 = SoftMoEPrompt's `route_rank`).
That is `score >= topk_score`. This mirrors the soft-MoE routing logic and
keeps exactly the experts the router would have selected for the old query.

Anchor positions (and the kept-expert set) are determined entirely from the
shared task-0 snapshot, so they are IDENTICAL across all 3 plots. Only the
red "drifted" positions differ per case.
"""
import argparse
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


def compute_per_expert(pk_dict, pv_dict, query_dict):
    """For each (layer, head): return raw routing scores (E,), softmax weights
    (E,) and per-expert contributions (E, D) under the given query."""
    out = {}
    for key in sorted(pk_dict.keys()):
        K = pk_dict[key]                       # (E, D)
        V = pv_dict[key]                       # (E, D)
        q = query_dict[key]                    # (1, D)
        scores = (q @ K.t()).squeeze(0)        # (E,)
        w = torch.softmax(scores, dim=-1)      # (E,)
        contribs = w.unsqueeze(-1) * V         # (E, D)
        out[key] = (scores.numpy(), w.numpy(), contribs.numpy())
    return out


def select_topk_experts(per_expert_anchor, topk):
    """Keep experts whose anchor routing score is in the top-K per (e, h).
    Returns: list of (e, h, l) tuples and numpy array (N_kept, D)."""
    kept_keys = []
    kept_anchor = []
    for (e, h), (scores, _, contribs) in per_expert_anchor.items():
        E = len(scores)
        k = min(max(int(topk), 1), E)
        topk_score = np.partition(scores, -k)[-k]   # k-th largest score
        for l in range(E):
            if scores[l] >= topk_score:
                kept_keys.append((e, h, l))
                kept_anchor.append(contribs[l])
    if not kept_anchor:
        return [], np.zeros((0, 64))
    return kept_keys, np.stack(kept_anchor, axis=0)


def stack_kept_contribs(per_expert_dict, kept_keys):
    """Given a per_expert dict and the list of kept (e, h, l) keys, stack the
    corresponding contribution vectors into (N_kept, D)."""
    rows = []
    for (e, h, l) in kept_keys:
        _, _, contribs = per_expert_dict[(e, h)]
        rows.append(contribs[l])
    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk", type=int, default=5,
        help="Keep experts whose anchor routing score is among the top-K "
             "per (layer, head). Default 5 matches SoftMoEPrompt's route_rank.",
    )
    args = parser.parse_args()
    K = args.topk

    # 1. Load shared anchor (pk_0, pv_0, query_0)
    if not os.path.exists(SHARED_SNAP):
        print(f"[fatal] shared snapshot not found: {SHARED_SNAP}")
        print("Run experiments/compare_losses.sh first to generate it.")
        return
    shared = torch.load(SHARED_SNAP, map_location="cpu")
    pk_0, pv_0, query_0 = shared["pk"], shared["pv"], shared["query"]

    per_expert_anchor = compute_per_expert(pk_0, pv_0, query_0)
    kept_keys, contrib_anchor = select_topk_experts(per_expert_anchor, K)

    if not kept_keys:
        print(f"No experts passed top-{K}. Snapshots look empty?")
        return
    n_total_experts = sum(len(v[0]) for v in per_expert_anchor.values())
    print(f"Top-{K} per (layer, head): kept {len(kept_keys)} / {n_total_experts} "
          f"experts ({len(kept_keys) / n_total_experts:.1%}).")

    # 2. Per-case drifted contributions on the SAME kept (e, h, l) set
    contrib_stars = {}
    for case, _ in CASES:
        snap1 = os.path.join(OUT_BASE, case, "snapshots", "snapshot_task1.pt")
        if not os.path.exists(snap1):
            print(f"[skip] {case}: missing {snap1}")
            continue
        s1 = torch.load(snap1, map_location="cpu")
        per_expert_star = compute_per_expert(s1["pk"], s1["pv"], query_0)
        contrib_stars[case] = stack_kept_contribs(per_expert_star, kept_keys)

    if not contrib_stars:
        print("No cases found, nothing to plot.")
        return

    # 3. Joint PCA — anchor positions identical across all plots
    union = np.concatenate(
        [contrib_anchor] + list(contrib_stars.values()),
        axis=0,
    )
    pca = PCA(n_components=2)
    pca.fit(union)
    proj_anchor = pca.transform(contrib_anchor)
    proj_stars = {c: pca.transform(v) for c, v in contrib_stars.items()}

    # Shared axis limits across plots
    all_proj = np.concatenate([proj_anchor] + list(proj_stars.values()), axis=0)
    pad_x = 0.05 * (all_proj[:, 0].max() - all_proj[:, 0].min() + 1e-9)
    pad_y = 0.05 * (all_proj[:, 1].max() - all_proj[:, 1].min() + 1e-9)
    xlim = (all_proj[:, 0].min() - pad_x, all_proj[:, 0].max() + pad_x)
    ylim = (all_proj[:, 1].min() - pad_y, all_proj[:, 1].max() + pad_y)

    # 4. Plot per case
    for case, title in CASES:
        if case not in contrib_stars:
            continue
        p1 = proj_anchor
        p2 = proj_stars[case]
        c_star = contrib_stars[case]

        l2_drift = np.linalg.norm(contrib_anchor - c_star, axis=1).mean()
        cos = (contrib_anchor * c_star).sum(axis=1) / (
            np.linalg.norm(contrib_anchor, axis=1)
            * np.linalg.norm(c_star, axis=1)
            + 1e-12
        )
        cossim = cos.mean()

        fig, ax = plt.subplots(figsize=(8, 6.5))
        for i in range(p1.shape[0]):
            ax.plot([p1[i, 0], p2[i, 0]], [p1[i, 1], p2[i, 1]],
                    color="gray", alpha=0.35, linewidth=0.6)
        ax.scatter(p1[:, 0], p1[:, 1], c="tab:blue", s=45, alpha=0.85,
                   label=r"$c^{old}_{e,h,l}$  (anchor task 0)")
        ax.scatter(p2[:, 0], p2[:, 1], c="tab:red", s=45, alpha=0.85,
                   marker="x",
                   label=r"$c^{cur}_{e,h,l}$  (after task 1)")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(
            f"{title}\n"
            f"top-{K} experts per (layer, head) = {len(kept_keys)} / {n_total_experts}   |   "
            f"mean L2 drift = {l2_drift:.4f}   |   mean cossim = {cossim:.4f}"
        )
        ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(OUT_BASE, f"attention_drift_{case}_top{K}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}   (drift={l2_drift:.4f}, cossim={cossim:.4f})")


if __name__ == "__main__":
    main()
