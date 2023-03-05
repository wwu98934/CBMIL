# Clustering-based Multi-instance Learning Network for Whole Slide Image Classification

This repo is the PyTorch implementation for the CBMIL described in the paper "Clustering-based Multi-instance Learning Network for Whole Slide Image Classification".

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
  --exist_ok \
  --save_model
```



### Data Organization

The format of  input csv file:

|  case_id   |           features_filepath            | label |            clusters_filepath            |          clusters_json_filepath          |
| :--------: | :------------------------------------: | :---: | :-------------------------------------: | :--------------------------------------: |
| normal_001 | /path/to/patch_features/normal_001.npz |   0   | /path/to/cluster_indices/normal_001.npz | /path/to/cluster_indices/normal_001.json |
|    ...     |                  ...                   |  ...  |                   ...                   |                   ...                    |

> **case_id**: [str] the index for each WSI. 
>
> **features_filepath**: [str] the .npz file path for each WSI, this .npz file contains several keywords as follows: 
>
> - filename: [str] case_id. 
> - img_features: [numpy.ndarray] the all patch's features as a numpy.ndarray, the shape is (num_patches, dim_features), like (1937, 512). 
>
> **label**: [int] the label of the WSI. 
>
> **clusters_filepath**: [str] this .npz file indicates the clustering category corresponding to each patch in WSI. It contaions several keywords as follows:
>
> - filename: [str] case_id.
> - features_cluster_indices: [numpy.ndarray] This array represents the clustering category of each patch feature in WSI, it's shape is (num_pathces, 1). 
>
> **clusters_json_filepath**: [str] This JSON file represents the patch index contained in each category of clustering, like:
>
> ```json
> [
>     [0, 30, 57, 58, 89, 113, 124, 131, ...],
>     [11, 13, 22, 25, 26, 34, 35, 45, 49, 50, 51, ...],
>     ...
>     [1, 8, 15, 16, ...]
> ]
> ```
>
> Each list represents a category.



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



