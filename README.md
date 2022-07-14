# Clustering-based Multi-instance Learning Network for Whole Slide Image Classification with Multi-task Learning

This repo is the PyTorch implementation for the MuRCL described in the paper "Clustering-based Multi-instance Learning Network for Whole Slide Image Classification with Multi-task Learning".

![pipline](figs/pipeline.png)

## Training

```shell
python train.py \
  --dataset CAMELYON16_20x_s256 \
  --data_csv /path/to/data_csv.csv \
  --data_split_json /path/to/data_split.json \
  --num_patches 1024 \
  --preload \
  --optimizer Adam \
  --scheduler CosineAnnealingLR \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0001 \
  --wdecay 0.0001 \
  --warmup 0 \
  --arch CBMIL \
  --alpha 0.8 \
  --temperature 1 \
  --num_classes 2 \
  --cl_weight 0.1 \
  --device 3 \
  --save_dir_flag cw_${CW}_lr_${LR}_wd_${WD} \
  --exist_ok \
  --save_model
```

## Visualization

### heatmaps

```shell
python create_heatmaps.py
```

### clusters  heatmap

```shell
python visualize_clusters_heatmaps.py
```

![heatmaps](figs/heatmaps.png)


