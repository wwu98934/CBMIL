import os
import cv2
import argparse
import openslide
import numpy as np
from pathlib import Path
from xml.dom import minidom
import torch

from dataset.datasets import ClusterFeaturesList
from utils.general import load_json
from models import cbmil


def get_datasets(args):
    train_set = ClusterFeaturesList(data_csv=args.data_csv, shuffle=False, preload=args.preload)
    return train_set, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    print(f"Creating model {args.arch}...")
    if args.arch == 'CBMIL':
        model = cbmil.CBMIL(
            dim_feature=dim_patch,
            num_sample_feature=1024
        )
    else:
        raise ValueError(f'args.arch error, {args.arch}. ')
    model = torch.nn.DataParallel(model).cuda()

    assert args.checkpoint is not None
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {msg.missing_keys}")

    assert model is not None, "creating model failed. "
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model


def get_three_points(x_step, y_step, size):
    top_left = (int(x_step * size), int(y_step * size))
    bottom_right = (int(top_left[0] + size), int(top_left[1] + size))
    center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    return top_left, bottom_right, center


def load_annotations_xml(annotations_xml):
    dom = minidom.parse(annotations_xml)
    root = dom.documentElement
    annotations = root.getElementsByTagName('Annotation')

    contours = []
    for a in annotations:
        coords = a.getElementsByTagName('Coordinates')[0].getElementsByTagName('Coordinate')
        contour = np.array([[c.getAttribute('X'), c.getAttribute('Y')] for c in coords], dtype=np.float)
        contour = np.expand_dims(contour, 1)
        contours.append(contour)
        # print(contour.shape)
    return contours


def create_heatmap(coord_filepath, attention, cluster_indices, slide_level=-1, contours=None):
    coord_dict = load_json(coord_filepath)
    slide_filepath = coord_dict['slide_filepath']
    num_row, num_col = coord_dict['num_row'], coord_dict['num_col']
    num_patches = coord_dict['num_patches']
    coords = coord_dict['coords']
    patch_size_level0 = coord_dict['patch_size_level0']
    slide = openslide.open_slide(slide_filepath)
    thumbnail = slide.get_thumbnail(slide.level_dimensions[slide_level]).convert('RGB')
    thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
    level_downsample = slide.level_downsamples[slide_level]
    assert num_patches == len(coords) == len(cluster_indices), f"{num_patches}-{len(coords)}-{len(cluster_indices)}"

    attention = np.uint8(255 * ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))))
    print(f"attention: {attention.shape}")
    colors = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
    print(f"colors: {colors.shape}")

    heatmap = np.ones(thumbnail.shape, dtype=np.uint8) * 255
    for i in range(num_patches):
        row, col = coords[i]['row'], coords[i]['col']
        points = get_three_points(col, row, patch_size_level0 / level_downsample)
        color_idx = cluster_indices[i]
        c = (int(colors[color_idx, 0, 0]), int(colors[color_idx, 0, 1]), int(colors[color_idx, 0, 2]))
        cv2.rectangle(heatmap, points[0], points[1], color=c, thickness=-1)

    heatmap = cv2.addWeighted(heatmap, 0.5, thumbnail, 0.5, 0)

    if contours is not None:
        contours = [np.asarray(c / level_downsample).astype(np.int32) for c in contours]
        heatmap = cv2.drawContours(heatmap, contours, -1, (0, 255, 255), thickness=5)
    return heatmap


def run(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # case_id_list = ['test_016']
    case_id_list = None

    dataset, dim_patch, data_size = get_datasets(args)
    model = create_model(args, dim_patch)
    model.eval()
    for cluster_feats, _, case_id in dataset:
        if case_id_list is not None and case_id not in case_id_list:
            continue

        heatmap_filepath = Path(args.save_dir) / f'{case_id}.png'
        if heatmap_filepath.exists() and not args.exist_ok:
            continue

        with torch.no_grad():
            cluster_feats = [feat.to(args.device) for feat in cluster_feats]
            cluster_attention = model.module.eval_forward(cluster_feats,
                                                          get_cluster_attention=True).cpu().numpy().reshape(-1)
        print(f"attention: {cluster_attention.shape}")

        coord_filepath = Path(args.coord_dir) / f'{case_id}.json'
        if not coord_filepath.exists():
            continue

        annotation_xml = Path(args.annotation_dir) / f'{case_id}.xml'
        if annotation_xml.exists() and args.draw_contours:
            contours = load_annotations_xml(str(annotation_xml))
        else:
            contours = None

        cluster_indices = np.load(str(Path(args.clusters_dir) / f'{case_id}.npz'))['features_cluster_indices'].reshape(
            -1)
        print(f"cluster_indices {cluster_indices.shape}:\n{cluster_indices}")

        heatmap = create_heatmap(str(coord_filepath), cluster_attention, cluster_indices, slide_level=args.slide_level,
                                 contours=contours)
        cv2.imwrite(str(heatmap_filepath), heatmap)
        print(f'{case_id} done!')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_csv', type=str,
                        default='/path/to/data_csv',
                        help='')
    parser.add_argument('--coord_dir', type=str, default='/dir/to/coord',
                        help='')
    parser.add_argument('--annotation_dir', type=str,
                        default='/dir/to/lesion_annotations',
                        help='')
    parser.add_argument('--clusters_dir', type=str,
                        default='/dir/to/cluster')
    parser.add_argument('--preload', action='store_true', default=False,
                        help="")
    # Architecture
    parser.add_argument('--arch', default='CBMIL', type=str, help='model name')
    parser.add_argument('--checkpoint', default=None, type=str)
    # Save
    parser.add_argument('--save_dir', type=str, default='/dir/to/save/results')
    parser.add_argument('--draw_contours', action='store_true', default=False)
    parser.add_argument('--slide_level', type=int, default=2)
    parser.add_argument('--exist_ok', action='store_true', default=False)
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
