
## Requirements

The reference environment is `conda env name: oneprompt` — Python 3.10, PyTorch 2.x with CUDA. Two ways to reproduce:

**Option 1 — Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate oneprompt
```

**Option 2 — pip only:**
```bash
pip install -r requirements.txt
```

Key packages: `torch`, `torchvision`, `timm==0.9.12`, `scikit-learn`, `numpy`, `pyyaml`, `pillow`, `opencv-python`, `pandas`, `openpyxl`, `matplotlib`.

---

## Datasets

Datasets are **not included** in the repo (`data/` is gitignored). Create a folder `data/` and download the three public datasets below into it:

```
data/
├── cifar-100-python/        # extracted from the CIFAR-100 tarball (or auto via torchvision)
├── CUB_200_2011/            # extracted from the Caltech tarball
└── imagenet-r/              # extracted from the Berkeley tarball
```

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

---

## Checkpoints

Pretrained ViT-Base/16 weights are also not included (`pretrained/*.bin` is gitignored). Create a folder `pretrained/` and download the backbones used in the configs:

- [Sup-21K](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/blob/main/pytorch_model.bin) → `pretrained/vit_base_patch16_224_augreg_in21k.bin`
- [Sup-1K](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/blob/main/pytorch_model.bin) → `pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin`
- [iBOT-1k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth) → `pretrained/ibot_vit_base16.pth`
- [DINO-1k](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) → `pretrained/dino_vitbase16_pretrain.pth`

Select one via `--pretrained_weight {sup21k|sup1k|ibot1k|dino1k}` (default `sup1k`).

---

## Training

Run the following under the project root. Scripts are set up for **1 GPU**.

```bash
bash experiments/cifar-100.sh
```

Results are saved under `outputs/` (also gitignored).

### Attention Preservation Loss

The `SoftMoEPrompt` learner accepts an additional `mu_attention` weight as the **8th** value of `--prompt_param`. When `> 0` and `task_id > 0`, the loss preserves the per-expert attention output under the previous task's mean reduced-query, weighted by a soft sigmoid importance gate so only experts active in the old task are constrained.

```
--prompt_param  prompt_length  route_rank  mu_router  mu_proto  eps  soft_alpha  soft_tau  mu_attention
                     50           5         1e-5      1e-5     1e-6     1.0       0.1        1e-3
```

Set `mu_attention=0` (or omit the 8th value) to disable; backwards-compatible with the original SoftMoE setup.

Sweep example for CIFAR-100:
```bash
for mu in 0 1e-5 1e-4 1e-3 1e-2; do
    python -u run.py --config configs/cifar-100_prompt_vapt.yaml \
        --gpuid 0 --repeat 1 --overwrite 1 \
        --learner_type prompt --learner_name SoftMoEPrompt \
        --prompt_param 50 5 1e-5 1e-5 1e-6 1.0 0.1 $mu --seeds 0 \
        --crct_epochs 50 --ca_batch_size_ratio 1 \
        --log_dir outputs/CIFAR100/10-task/soft-moe-attn-${mu} \
        --adaptive_pred
done
```

### Comparing loss configurations

`experiments/compare_losses.sh` runs a two-phase experiment that trains task 0 once (shared anchor) and then trains task 1 three times under three loss configurations (no extra losses / proto only / attention only). After training it calls `experiments/visualize_attention_drift.py` to render PCA scatter plots of `attention_0` vs `attention_0*` (the same query reused with task-1 prompts) into `outputs/compare_losses/attention_drift_*.png`.

```bash
rm -rf outputs/compare_losses     # start fresh
bash experiments/compare_losses.sh
```

---

## Reference Codes

[1] [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)

[2] [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)

[3] [VQ-Prompt](https://github.com/jiaolifengmi/VQ-Prompt)
