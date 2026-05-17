"""
Count parameters, train GFLOPs, and test GFLOPs of the SoftMoE prompt model.

Usage:
    python utils/count_flops.py
    python utils/count_flops.py --num_classes 200 --topk -1     # CUB-200, soft routing
    python utils/count_flops.py --prompt_flag soft-moe --batch_size 1
"""
import argparse
import os
import sys
import torch

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.zoo import ViTZoo


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    prompt = sum(p.numel() for n, p in model.named_parameters() if "prompt" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if "last" in n)
    backbone = total - prompt - head
    # In SoftMoE, optimizer only updates prompt + head (backbone frozen by
    # construction in the optimizer init).
    actually_trained = prompt + head
    return {
        "total (model)": total,
        "backbone (frozen at training)": backbone,
        "prompt (trained)": prompt,
        "head (trained)": head,
        "trained per task = prompt + head": actually_trained,
    }


def fmt(n):
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def count_macs_with_thop(model, x, train=False):
    """Returns MACs (multiply-accumulate ops). Most ViT papers report this as 'GFLOPs'."""
    from thop import profile

    model.eval() if not train else model.train()

    class Wrapper(torch.nn.Module):
        def __init__(self, m, train_flag):
            super().__init__()
            self.m = m
            self.train_flag = train_flag

        def forward(self, x):
            return self.m(x, train=self.train_flag)

    wrapped = Wrapper(model, train)
    macs, _ = profile(wrapped, inputs=(x,), verbose=False)
    return macs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", type=int, default=100,
                   help="CIFAR-100=100, CUB-200=200, ImageNet-R=200")
    p.add_argument("--prompt_flag", type=str, default="soft-moe",
                   choices=["soft-moe", "vapt", "baseline_smope"])
    p.add_argument("--prompt_length", type=int, default=50,
                   help="2 * num_experts")
    p.add_argument("--n_tasks", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    # SoftMoE prompt_param: [length, route_rank, mu_router, mu_proto, eps,
    #                       soft_alpha, soft_tau, mu_attention]
    prompt_param = [args.n_tasks,
                    [args.prompt_length, 5, 1e-5, 1e-5, 1e-6, 1.0, 0.1, 1e-3]]

    print(f"Building ViTZoo: prompt_flag={args.prompt_flag}, "
          f"num_classes={args.num_classes}, n_tasks={args.n_tasks}")

    model = ViTZoo(
        num_classes=args.num_classes,
        pt=True,                  # build ViT skeleton; pretrained=None skips weight load
        prompt_flag=args.prompt_flag,
        prompt_param=prompt_param,
        pretrained=None,
    )
    model.task_id = 0
    model = model.to(args.device).eval()

    # ---------- Parameters ----------
    print("\n" + "=" * 60)
    print("PARAMETERS")
    print("=" * 60)
    params = count_params(model)
    for k, v in params.items():
        print(f"  {k:30s}: {fmt(v):>14s}  ({v:,})")

    # ---------- FLOPs ----------
    x = torch.randn(args.batch_size, 3, 224, 224, device=args.device)

    print("\n" + "=" * 60)
    print(f"COMPUTE COST (per image, batch_size={args.batch_size})")
    print("=" * 60)
    print("  Convention 1: GFLOPs = MACs (most ViT/CL papers)")
    print("  Convention 2: GFLOPs = 2 x MACs (strict floating-point ops)")
    print()

    macs_test = count_macs_with_thop(model, x, train=False)
    print(f"  TEST (forward only)")
    print(f"    GMACs   : {macs_test / 1e9:>8.3f}  (paper-style 'GFLOPs')")
    print(f"    GFLOPs  : {2 * macs_test / 1e9:>8.3f}  (strict)")
    print()

    macs_fwd_train = count_macs_with_thop(model, x, train=True)
    print(f"  TRAIN (forward only, train=True branch)")
    print(f"    GMACs   : {macs_fwd_train / 1e9:>8.3f}")
    print(f"    GFLOPs  : {2 * macs_fwd_train / 1e9:>8.3f}")
    print()

    # Convention: backward ≈ 2 x forward → total ≈ 3 x forward
    print(f"  TRAIN (forward + backward, ~3x forward)")
    print(f"    GMACs   : {macs_fwd_train * 3 / 1e9:>8.3f}")
    print(f"    GFLOPs  : {2 * macs_fwd_train * 3 / 1e9:>8.3f}")
    print()


if __name__ == "__main__":
    main()
