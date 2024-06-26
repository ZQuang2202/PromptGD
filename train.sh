python3 train_network_clip.py \
    --network grconvnet3 \
    --use-depth 0 \
    --dataset grasp-anything \
    --dataset-path /home/bdi/AdvancedLiterateMachinery/DocumentUnderstanding/CLIP_OCR/Dataset/grasp-anything++/seen \
    --batch-size 4 \
    --batches-per-epoch 600 \
    --epochs 100 \
    --optim adam \
    --lr-step-size 10 \
    --logdir logs/ \
    --seen 1
