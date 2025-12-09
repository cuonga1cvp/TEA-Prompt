#  One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning

Official PyTorch code for "One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning".


## Requirements
Create a conda environment and install the following packages:
 * python=3.8.18
 * torch=2.0.0+cu118
 * torchvision=0.15.1+cu118
 * timm=0.9.12
 * scikit-learn=1.3.2
 * numpy
 * pyaml
 * pillow
 * opencv-python
 * pandas
 * openpyxl (write results to a xlsx file)
or simply run:
```setup
pip install -r requirements.txt
```
 
## Datasets
 * Create a folder `data/`
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

## Checkpoints
 * Create a folder `pretrained/`
 - [Sup-21K](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/blob/main/pytorch_model.bin)
 - [Sup-1K](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/blob/main/pytorch_model.bin)
 - [iBOT-1k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
 - [DINO-1k](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)  

## Training
Run the following commands under the project root directory. **The scripts are set up for 1 GPUs**.

```bash
sh experiments/cifar-100.sh
sh experiments/imagenet-r_all.sh
sh experiments/cub-200.sh
```

## Results
Results will be saved in a folder named `outputs/`.


## Reference Codes
[1] [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)

[2] [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)

[3] [VQ-Prompt](https://github.com/jiaolifengmi/VQ-Prompt)