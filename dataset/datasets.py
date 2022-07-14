import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterable, Dict, Union, List

import torch
from torch.utils.data.dataset import Dataset


def load_json(filename):
    with open(filename) as fp:
        data_dict = json.load(fp)
    return data_dict


class WSIDataset(Dataset):
    """Basic WSI Dataset, which can obtain the features of each patch of WSIs."""

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: Union[int, float, None] = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True) -> None:
        """Initialization constructor.

        :param str or Path data_csv: A CSV file's filepath for organization WSI data, as detailed in our README
        :param Iterable[str] indices: A list containing the specified `case_id`, if None, fetching all `case_id` in the `data_csv` file
        :param int num_sample_patches: The number of sampled patches, if None, the value is the number of all patches
        :param bool fixed_size: If True, the size of the number of patches is fixed
        :param bool shuffle: If True, shuffle the order of all WSIs
        :param bool patch_random: If True, shuffle the order of patches within a WSI during reading this dataset
        :param bool preload: If True, all feature files are loaded at initialization
        """
        super(WSIDataset, self).__init__()
        self.data_csv = data_csv
        self.indices = indices
        self.num_sample_patches = num_sample_patches
        self.fixed_size = fixed_size
        self.preload = preload
        self.patch_random = patch_random

        self.samples = self.process_data()
        if self.indices is None:
            self.indices = self.samples.index.values
        if shuffle:
            self.shuffle()

        self.patch_dim = np.load(self.samples.at[self.samples.index[0], 'features_filepath'])['img_features'].shape[-1]

        if self.preload:
            self.patch_features = self.load_patch_features()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.indices[index]

        if self.preload:
            patch_feature = self.patch_features[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']

        patch_feature = self.sample_feat(patch_feature)
        if self.fixed_size:
            patch_feature = self.fix_size(patch_feature)
        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, label, case_id

    def shuffle(self) -> None:
        """Shuffle the order of WSIs. """
        random.shuffle(self.indices)

    def process_data(self):
        """Load the `data_csv` file by `indices`. """
        data_csv = pd.read_csv(self.data_csv)
        data_csv.set_index(keys='case_id', inplace=True)
        if self.indices is not None:
            samples = data_csv.loc[self.indices]
        else:
            samples = data_csv
        return samples

    def load_patch_features(self) -> Dict[str, np.ndarray]:
        """Load the all the patch features of all WSIs. """
        patch_features = {}
        for case_id in self.indices:
            patch_features[case_id] = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
        return patch_features

    def sample_feat(self, patch_feature: np.ndarray) -> np.ndarray:
        """Sample features by `num_sample_patches`. """
        num_patches = patch_feature.shape[0]
        if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
            sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
            sample_indices = sorted(sample_indices)
            patch_feature = patch_feature[sample_indices]
        if self.patch_random:
            np.random.shuffle(patch_feature)
        return patch_feature

    def fix_size(self, patch_feature: np.ndarray) -> np.ndarray:
        """Fixed the shape of each WSI feature. """
        if patch_feature.shape[0] < self.num_sample_patches:
            margin = self.num_sample_patches - patch_feature.shape[0]
            feat_pad = np.zeros(shape=(margin, self.patch_dim))
            feat = np.concatenate((patch_feature, feat_pad))
        else:
            feat = patch_feature[:self.num_sample_patches]
        return feat


class WSIWithCluster(WSIDataset):
    """A WSI Dataset with its cluster result"""

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: Union[int, float, None] = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True) -> None:
        """Initialization constructor.

        :param str or Path data_csv: A CSV file's filepath for organization WSI data, the end of the filename must provide the number of clusters, as detailed in our README
        :param Iterable[str] indices: A list containing the specified `case_id`, if None, fetching all `case_id` in the `data_csv` file
        :param int num_sample_patches: The number of sampled patches, if None, the value is the number of all patches
        :param bool fixed_size: If True, the size of the number of patches is fixed
        :param bool shuffle: If True, shuffle the order of all WSIs
        :param bool patch_random: If True, shuffle the order of patches within a WSI during reading this dataset
        :param bool preload: If True, all feature files are loaded at initialization
        """
        super(WSIWithCluster, self).__init__(data_csv, indices, num_sample_patches, fixed_size, shuffle, patch_random,
                                             preload)
        # The filename of `data_csv` must provide the number of clusters at the end.
        # eg. camelyon16_10.csv, the 10 indicates the number of clusters.
        self.num_clusters = int(Path(data_csv).stem.split('_')[-1])

        if self.preload:
            self.cluster_indices = self.load_cluster_indices()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, str]:
        """Return the WSI features, cluster indices, label, and the case_id. """
        case_id = self.indices[index]

        if self.preload:
            patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
            cluster_indices = load_json(self.samples.at[case_id, 'clusters_json_filepath'])

        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, cluster_indices, label, case_id

    def load_cluster_indices(self) -> Dict[str, List[List[int]]]:
        cluster_indices = {}
        for case_id in self.indices:
            cluster_indices[case_id] = load_json(self.samples.at[case_id, 'clusters_json_filepath'])
        return cluster_indices


class ClusterFeaturesList(WSIWithCluster):
    """A WSI Dataset, which patches sampled by cluster result. """

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: int = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True,
                 half_split: bool = False,
                 subset: bool = False) -> None:
        super(ClusterFeaturesList, self).__init__(data_csv, indices, num_sample_patches, fixed_size, shuffle,
                                                  patch_random, preload)
        self.half_split = half_split
        self.subset = subset
        assert not (self.half_split and self.subset)

    def __getitem__(self, index: int):
        case_id = self.indices[index]

        if self.preload:
            patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
            cluster_indices = load_json(self.samples.at[case_id, 'clusters_json_filepath'])
        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)

        if self.half_split:
            half1, half2 = self.random_split_half(cluster_indices)
            patch_feature1 = self.compose_features(patch_feature, half1)
            patch_feature2 = self.compose_features(patch_feature, half2)
            return patch_feature1, patch_feature2, label, case_id
        elif self.subset:
            all_patch_feature = self.compose_features(patch_feature, cluster_indices)
            subset_indices = self.random_sample_subset(cluster_indices)
            # for i, sb in enumerate(subset_indices):
            #     print(f"c{i} {len(sb)}:\n{sb}")
            sub_patch_feature = self.compose_features(patch_feature, subset_indices)
            # for x, y in zip(all_patch_feature, sub_patch_feature):
            #     print(f"{x.shape} | {y.shape}")
            return all_patch_feature, sub_patch_feature, label, case_id
        else:
            patch_feature = self.compose_features(patch_feature, cluster_indices)
            return patch_feature, label, case_id

    @staticmethod
    def compose_features(patch_feature: np.ndarray, cluster_indices: List[List[int]]) -> List:
        """Compose by cluster indices. """
        cluster_features = []
        for cluster_index in cluster_indices:
            assert len(cluster_index) != 0, f'{cluster_index}'

            feat = patch_feature[cluster_index]
            feat = torch.as_tensor(feat)
            cluster_features.append(feat)
        return cluster_features

    @staticmethod
    def random_split_half(cluster_indices: List[List[int]]):
        half1, half2 = [], []
        for cluster_index in cluster_indices:
            if len(cluster_index) == 1:
                half1.append(cluster_index)
                half2.append(cluster_index)
                continue
            mid = len(cluster_index) // 2
            np.random.shuffle(cluster_index)
            half1.append(cluster_index[:mid])
            np.random.shuffle(cluster_index)
            half2.append(cluster_index[:mid])
        return half1, half2

    def random_sample_subset(self, cluster_indices: List[List[int]]):
        num_patches = sum([len(c) for c in cluster_indices])
        if self.num_sample_patches is None:
            sample_ratio = 1.
        else:
            sample_ratio = self.num_sample_patches / num_patches
        sample_indices = []
        if sample_ratio < 1:
            for c in range(self.num_clusters):
                num_patch_c = len(cluster_indices[c])
                size = int(np.rint(num_patch_c * sample_ratio))
                if size == 0:
                    size = 1
                sample = np.random.choice(num_patch_c, size=size, replace=False)
                sample_indices.append([cluster_indices[c][s] for s in sample])
        else:
            sample_indices = cluster_indices
        return sample_indices
