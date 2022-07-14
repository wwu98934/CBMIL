import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .mil import CLAM_SB
from typing import Tuple, Iterable, Dict, Union, List


class NonLocalRanking(nn.Module):
    def __init__(self, in_features, dropout_v=0.0):
        super(NonLocalRanking, self).__init__()
        self.q = nn.Linear(in_features, 128)

        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(in_features, in_features)
        )

    def forward(self, feats, key_feat, top_k=None, top_fuse=False, attention_only=False):
        # print(f"feats: {feats.shape} | key_feat: {key_feat.shape}")
        device = feats.device
        Q = self.q(feats).view(feats.shape[0], -1)

        q_key = self.q(key_feat)
        A = torch.mm(Q, q_key.transpose(0, 1))
        if attention_only:
            return A
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)

        if top_k is not None:
            assert top_k <= A.shape[0], f"{top_k} <= {A.shape[0]}?"
            _, top_k_index = torch.topk(A, top_k, dim=0)
            top_k_features = torch.index_select(feats, dim=0, index=torch.flatten(top_k_index))

            if top_fuse:
                V = self.v(top_k_features)
                A_ = torch.index_select(A, dim=0, index=torch.flatten(top_k_index))
                # TODO: softmax
                A_ = F.softmax(A_ / torch.sqrt(torch.tensor(V.shape[1], dtype=torch.float32, device=device)), 0)
                fusion_feature = torch.mm(A_.transpose(0, 1), V)  # 1xN * N*dim
            else:
                V = self.v(feats)  # N x dim
                fusion_feature = torch.mm(A.transpose(0, 1), V)  # 1xN * N*dim
            return top_k_features, fusion_feature
        else:
            V = self.v(feats)  # N x dim
            fusion_feature = torch.mm(A.transpose(0, 1), V)  # 1xN * N*dim
            return feats, fusion_feature


class AdaptiveSelection(nn.Module):
    def __init__(self, dim_feature, num_sample_feature=1024):
        super(AdaptiveSelection, self).__init__()
        self.num_sample_feature = num_sample_feature
        self.ranking = NonLocalRanking(in_features=dim_feature)

    def forward(self, cluster_features, key_feats, selecting='adaptive', top_fuse=False):
        assert selecting in ['all', 'cluster_random', 'adaptive'], f"Selecting method error: {selecting}"
        num_patches = sum([c.shape[0] for c in cluster_features])
        sample_ratio = self.num_sample_feature / num_patches

        selected_feature_list, cluster_fusion_feature_list = [], []
        for cluster_feature, key_feat in zip(cluster_features, key_feats):
            if sample_ratio < 1:
                sample_size = int(np.rint(cluster_feature.shape[0] * sample_ratio))
            else:
                sample_size = cluster_feature.shape[0]

            if selecting == 'all':
                selected_feature, fusion_feature = self.ranking(cluster_feature, key_feat, top_k=None)
            elif selecting == 'cluster_random':
                rand_idx = torch.randperm(cluster_feature.shape[0], device=cluster_feature.device)
                selected_feature = cluster_feature[rand_idx[:sample_size]]
                selected_feature, fusion_feature = self.ranking(selected_feature, key_feat, top_k=None)
            elif selecting == 'adaptive':
                selected_feature, fusion_feature = self.ranking(cluster_feature, key_feat, top_k=sample_size,
                                                                top_fuse=top_fuse)
            else:
                raise ValueError(
                    f"selecting method error: {selecting}, expected in {['all', 'cluster_random', 'adaptive']}"
                )

            selected_feature_list.append(selected_feature)
            cluster_fusion_feature_list.append(fusion_feature)
        selected_features = torch.cat(selected_feature_list, dim=0)
        cluster_fusion_feature = torch.cat(cluster_fusion_feature_list, dim=0)

        # fixing feature size
        if selected_features.shape[0] < self.num_sample_feature:
            margin = self.num_sample_feature - selected_features.shape[0]
            feat_pad = torch.zeros(size=(margin, selected_features.shape[1]), device=selected_features.device)
            selected_features = torch.cat((selected_features, feat_pad))
        else:
            selected_features = selected_features[:self.num_sample_feature]

        return selected_features, cluster_fusion_feature


class PPMIL(nn.Module):
    def __init__(self, dim_feature, dim_embedding=512, dim_output=2):
        super(PPMIL, self).__init__()
        self.patch_mil = CLAM_SB(
            gate=True,
            dropout=True,
            n_classes=dim_output,
            subtyping=True,
            in_dim=dim_feature
        )
        self.patch_mil.classifiers = nn.Identity()

        # gated attention for phenotype-level features
        self.embed = nn.Sequential(
            nn.Linear(dim_feature, dim_embedding),
            nn.ReLU()
        )
        self.cluster_attention_V = nn.Sequential(
            nn.Linear(dim_embedding, 256),
            nn.Tanh()
        )
        self.cluster_attention_U = nn.Sequential(
            nn.Linear(dim_embedding, 256),
            nn.Sigmoid()
        )
        self.cluster_attention_weights = nn.Linear(256, 1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=dim_embedding * 2, out_features=dim_embedding),
            nn.ReLU(),
            nn.Linear(in_features=dim_embedding, out_features=dim_output)
        )

    def forward(self, selected_features, cluster_fusion_features, label=None, attention_only=False):
        total_inst_loss = None
        if label is not None:
            _, mil_feature, total_inst_loss = self.patch_mil(selected_features, label)
        else:
            _, mil_feature = self.patch_mil(selected_features, label)

        H = self.embed(cluster_fusion_features)
        A_V = self.cluster_attention_V(H)  # NxD
        A_U = self.cluster_attention_U(H)  # NxD
        A = self.cluster_attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        A = A / math.sqrt(A.shape[-1])
        M = torch.mm(A, H)  # KxL

        fusion_feature = torch.cat((mil_feature, M), dim=1)
        output = self.fc(fusion_feature)

        if label is None:
            return output, fusion_feature
        else:
            return output, fusion_feature, total_inst_loss


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    @staticmethod
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward_diff(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if not z_i.shape[0] == self.batch_size:
            return self.forward_diff(z_i, z_j)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class CBMIL(nn.Module):
    def __init__(self, dim_feature, num_sample_feature=1024, dim_output=2, dim_cl=128, mixup_alpha=None,
                 bach_size=128, temperature=0.7, cl_weight=0.1, selecting='adaptive', top_fuse=False):
        super(CBMIL, self).__init__()
        self.mixup_alpha = mixup_alpha
        self.selecting = AdaptiveSelection(dim_feature=dim_feature, num_sample_feature=num_sample_feature)
        self.pp_mil = PPMIL(dim_feature=dim_feature, dim_output=dim_output)
        self.contrastive_fc = nn.Sequential(
            nn.Linear(in_features=dim_feature * 2, out_features=dim_feature),
            nn.ReLU(),
            nn.Linear(in_features=dim_feature, out_features=dim_cl)
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.cl_loss = NT_Xent(batch_size=bach_size, temperature=temperature)
        self.cl_weight = cl_weight
        self.select_method = selecting
        self.top_fuse = top_fuse

    def forward(self, inputs, label, train=True):
        if train:
            assert len(inputs) == 2, f'the number of views is {len(inputs)}'
            return self.train_forward(inputs, label)
        else:
            return self.eval_forward(inputs, label)

    def train_forward(self, cluster_features_views, labels):
        outputs, cl_feature0, ins_loss = self.batch_forward(cluster_features_views[0], labels=labels,
                                                            selecting=self.select_method, top_fuse=self.top_fuse)
        cl_feature1 = self.batch_forward(cluster_features_views[1], selecting='all', top_fuse=False)
        bag_loss = self.ce_loss(outputs, labels)
        ce_loss = 0.7 * bag_loss + 0.3 * ins_loss
        cl_loss = self.cl_loss(cl_feature0, cl_feature1)
        final_loss = self.cl_weight * cl_loss + (1 - self.cl_weight) * ce_loss
        return final_loss, outputs

    def batch_forward(self, batch_cluster_features, labels=None, selecting=None, top_fuse=None):
        batch_selected_features, batch_cluster_fusion_features = [], []
        for cluster_features in batch_cluster_features:
            key_feats = []
            for cluster_feature in cluster_features:
                with torch.no_grad():
                    _, key_feat = self.pp_mil.patch_mil(cluster_feature)
                    key_feats.append(key_feat)
            selected_features, cluster_fusion_features = self.selecting(cluster_features, key_feats,
                                                                        selecting=selecting, top_fuse=top_fuse)
            batch_selected_features.append(selected_features)
            batch_cluster_fusion_features.append(cluster_fusion_features)
        batch_selected_features = torch.stack(batch_selected_features, dim=0)
        batch_cluster_fusion_features = torch.stack(batch_cluster_fusion_features, dim=0)

        if self.mixup_alpha is not None:
            batch_selected_features = mixup(batch_selected_features, self.mixup_alpha)[0]

        if labels is not None:
            output_list, fusion_feature_list, ins_loss_list = [], [], []
            for selected_features, cluster_fusion_features, label in zip(batch_selected_features,
                                                                         batch_cluster_fusion_features, labels):
                output, fusion_feature, ins_loss = self.pp_mil(selected_features, cluster_fusion_features, label)
                output_list.append(output)
                fusion_feature_list.append(fusion_feature)
                ins_loss_list.append(ins_loss)
            outputs = torch.cat(output_list)
            fusion_features = torch.cat(fusion_feature_list)
            cl_feature = self.contrastive_fc(fusion_features)
            ins_loss = sum(ins_loss_list) / len(ins_loss_list)
            return outputs, cl_feature, ins_loss
        else:
            fusion_feature_list = []
            for selected_features, cluster_fusion_features in zip(batch_selected_features,
                                                                  batch_cluster_fusion_features):
                _, fusion_feature = self.pp_mil(selected_features, cluster_fusion_features)
                fusion_feature_list.append(fusion_feature)
            # outputs = torch.cat(output_list)
            fusion_features = torch.cat(fusion_feature_list)
            cl_feature = self.contrastive_fc(fusion_features)
            return cl_feature

    def eval_forward(self, cluster_features, label=None, get_cluster_attention=False):
        key_feats = []
        for cluster_feature in cluster_features:
            with torch.no_grad():
                _, key_feat = self.pp_mil.patch_mil(cluster_feature)
                key_feats.append(key_feat)
        selected_features, cluster_fusion_features = self.selecting(cluster_features, key_feats)
        if get_cluster_attention:
            return self.pp_mil(selected_features, cluster_fusion_features, label, attention_only=True)
        if label is None:
            output, *_ = self.pp_mil(selected_features, cluster_fusion_features, label)
            return output
        else:
            output, _, total_inst_loss = self.pp_mil(selected_features, cluster_fusion_features, label)
            return output, total_inst_loss


def mixup(inputs: torch.Tensor, alpha: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix-up a batch tentor. """
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx

# if __name__ == '__main__':
#     model = AdaptiveGatedMIL(dim_feature=512, num_sample_feature=100, batch=True, mixup_alpha=0.9)
#     num_patches_list = [231, 321, 433, 43, 312, 54, 54, 12, 69, 98]
#     inputs = [[torch.randn(size=(n, 512)) for n in num_patches_list] for _ in range(16)]
#
#     key_feat = torch.randn(size=(1, 512))
#     output = model(inputs)
#     print(f"output:{output.shape}\n{output}")
