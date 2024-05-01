# Introduction
Implementation code for PromptGD - A language-driven Grasp Detection.

## Installation
- Create environment:
```bash
git clone https://github.com/ZQuang2202/PromptGD.git & cd PromptGD
conda create -n grasp python=3.9
conda activate grasp
```
- Install packages:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Datasets
- [Grasp-Anything](https://grasp-anything-2023.github.io/)

## Training
- Train baseline GR-ConvNet model:
```bash
python3 train_network_baseline.py \
    --network grconvnet3 \
    --use-depth 0 \
    --dataset grasp-anything \
    --dataset-path $path_to_your_grasp_anything_dataset \
    --batch-size 32 \
    --batches-per-epoch 100 \
    --epochs 100 \
    --optim adam \
    --lr 0.001 \
    --lr-step-size 10 \
    --logdir logs/ \
    --seen 1
```
- Train GR-ConvNet when augmented with instructional text:
```bash
python3 train_network_with_clip.py \
    --network grconvnet3 \
    --use-depth 0 \
    --dataset grasp-anything \
    --dataset-path $path_to_your_grasp_anything_dataset \
    --batch-size 8 \
    --batches-per-epoch 600 \
    --epochs 100 \
    --optim adam \
    --lr 0.001 \
    --lr-step-size 10 \
    --logdir logs/ \
    --seen 1
```
- Train PromptGD:
```bash
python3 train_network_PromptGD.py \
    --clip-version ViT-B/32 \
    --use-depth 0 \
    --dataset grasp-anything \
    --dataset-path $path_to_your_grasp_anything_dataset \
    --batch-size 8 \
    --batches-per-epoch 300 \
    --epochs 100 \
    --lr 0.003 \
    --lr-step-size 5 \
    --logdir logs/prompt_gd \
    --seen 1
```
## Testing
For testing procedure, we can apply the similar commands to test different baselines on different datasets:
```bash
python3 evaluate.py \
    --network $path_to_your_check_point \
    --input-size 224 \
    --dataset grasp-anything \
    --dataset-path  $path_to_your_grasp_anything_dataset \
    --use-depth 0 \
    --use-rgb 1 \
    --num-workers 8 \
    --n-grasp 1 \
    --iou-threshold 0.25 \
    --iou-eval \
    --seen 0
```
- I also provided logs training and pretrained weight of all model at https://huggingface.co/datasets/QuangNguyen22/PromptGD/tree/main
## Acknowledgement
The code is developed based on [Kumra et al.](https://github.com/skumra/robotic-grasping).
