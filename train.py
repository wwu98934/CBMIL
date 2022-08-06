import os
import copy
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn, optim

from models.cbmil import CBMIL
from dataset.datasets import ClusterFeaturesList
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, accuracy, init_seeds, \
    load_json, get_metrics, get_score


def create_save_dir(args):
    dir1 = f'{args.dataset}'
    dir2 = f'np_{args.num_patches}'
    dir3 = args.arch
    if args.arch in ['CBMIL']:
        arch_setting = [f'default']
    else:
        raise ValueError()
    dir4 = '_'.join(arch_setting)
    dir5 = "exp"
    if args.save_dir_flag is not None:
        dir5 = f'{dir5}_{args.save_dir_flag}'
    dir6 = f'seed{args.seed}'
    dir7 = f'fold{args.fold_idx}' if args.fold_idx is not None else None
    if dir7 is None:
        args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4 / dir5 / dir6)
    else:
        args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7)
    print(f"save_dir: {args.save_dir}")


def get_datasets(args):
    indices = load_json(args.data_split_json)
    if args.fold_idx is not None:
        indices = indices[args.fold_idx]
    print(f"train_data: {args.train_data}")

    train_set = ClusterFeaturesList(
        args.data_csv,
        indices=indices['train'],
        shuffle=True,
        patch_random=False,
        preload=args.preload,
        subset=True
    )
    valid_set = ClusterFeaturesList(
        args.data_csv,
        indices=indices['valid'],
        shuffle=False,
        preload=args.preload
    )
    test_set = ClusterFeaturesList(
        args.data_csv,
        indices=indices['test'],
        shuffle=False,
        preload=args.preload
    )
    return {'train': train_set, 'valid': valid_set, 'test': test_set}, train_set.patch_dim


def create_model(args, dim_patch):
    print(f"Creating model {args.arch}...")

    model = CBMIL(
        dim_feature=dim_patch,
        dim_output=args.num_classes,
        num_sample_feature=args.num_patches,
        mixup_alpha=args.alpha,
        bach_size=args.batch_size,
        temperature=args.temperature,
        cl_weight=args.cl_weight,
        selecting=args.selecting,
        top_fuse=True
    )

    model = torch.nn.DataParallel(model).cuda()

    assert model is not None, "creating model failed. "
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model


def get_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"args.loss error, error value is {args.loss}.")
    return criterion


def get_optimizer(args, model):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wdecay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    elif args.optimezer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.wdecay)
    else:
        raise ValueError('Optimizer not found. Accepted "Adam", "SGD"')
    return optimizer


def get_scheduler(args, optimizer):
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError("Optimizer not found. Accepted 'Adam' or 'SGD'")
    return scheduler


def train_CBMIL(args, epoch, train_set, model, criterion, optimizer, scheduler):
    print(f"training...")
    assert not args.batch_size == 1

    train_set.shuffle()
    losses, accs, batch_idx = AverageMeter(), AverageMeter(), 0
    progress_bar = tqdm(range(len(train_set)))
    labels_list, outputs_list = [], []
    x1_list, x2_list, label_list, step = [], [], [], 0
    for data_idx in progress_bar:
        cluster_feats1, cluster_feats2, label, _ = train_set[data_idx % args.train_size]
        cluster_feats1 = [feat.to(args.device) for feat in cluster_feats1]
        cluster_feats2 = [feat.to(args.device) for feat in cluster_feats2]
        label = label.unsqueeze(0).to(args.device)
        x1_list.append(cluster_feats1)
        x2_list.append(cluster_feats2)
        label_list.append(label)

        step += 1
        if step == args.batch_size or data_idx == len(train_set) - 1:
            labels = torch.cat(label_list)
            # print(labels.shape)
            loss, outputs, *_ = model([x1_list, x2_list], label=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            losses.update(loss.item())
            acc = accuracy(outputs.detach(), labels, topk=(1,))[0]
            accs.update(acc.item())

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            batch_idx += 1
            x1_list, x2_list, label_list, step = [], [], [], 0
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.train_size:3}. "
                f"LR: {scheduler.get_last_lr()[0] if scheduler is not None else args.lr:.6f}. "
                f"Loss: {losses.avg:.4f}. Acc: {accs.avg:.4f}"
            )
            progress_bar.update()
    progress_bar.close()
    if scheduler is not None and epoch >= args.warmup:
        scheduler.step()
    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses.avg, acc, auc, precision, recall, f1_score


def test_CBMIL(args, test_set, model, criterion, mode):
    with torch.no_grad():
        bag_output_list, label_list, case_id_list = [], [], []
        losses = AverageMeter()
        progress_bar = tqdm(test_set)
        for data_idx, (cluster_feats, labels, case_id) in enumerate(progress_bar):
            cluster_feats = [feat.to(args.device) for feat in cluster_feats]
            labels = labels.unsqueeze(0).to(args.device)

            outputs, ins_loss = model(cluster_feats, label=labels, train=False)

            bag_loss = criterion(outputs, labels)
            loss = 0.7 * bag_loss + 0.3 * ins_loss
            losses.update(loss.item())

            label_list.append(labels)
            bag_output_list.append(outputs)
            case_id_list.append(case_id)

            progress_bar.set_description(f"{mode} Iter: {data_idx + 1:2}/{len(test_set):2}. ")
        # end Batch ----------------------------------------------------------------------------------------------------
        progress_bar.close()

        bag_outputs_tensor = torch.cat(bag_output_list).view(len(label_list), -1)
        labels_tensor = torch.cat(label_list)

        acc, auc, precision, recall, f1_score = get_metrics(bag_outputs_tensor, labels_tensor)

    return loss.item(), acc, auc, precision, recall, f1_score, bag_outputs_tensor, labels_tensor, case_id_list


# Basic Functions ------------------------------------------------------------------------------------------------------
def train(args, train_set, valid_set, test_set, model, criterion, optimizer, scheduler, save_dir):
    # Init variables
    best_train_acc = BestVariable(order='max')
    best_valid_acc = BestVariable(order='max')
    best_test_acc = BestVariable(order='max')
    best_train_auc = BestVariable(order='max')
    best_valid_auc = BestVariable(order='max')
    best_test_auc = BestVariable(order='max')
    best_train_loss = BestVariable(order='min')
    best_valid_loss = BestVariable(order='min')
    best_test_loss = BestVariable(order='min')
    best_score = BestVariable(order='max')
    final_loss, final_acc, final_auc, final_precision, final_recall, final_f1_score, final_epoch = 0., 0., 0., 0., 0., 0., 0
    header = ['epoch', 'train', 'valid', 'test', 'best_train', 'best_valid', 'best_test']
    losses_csv = CSVWriter(filename=Path(save_dir) / 'losses.csv', header=header)
    accs_csv = CSVWriter(filename=Path(save_dir) / 'accs.csv', header=header)
    aucs_csv = CSVWriter(filename=Path(save_dir) / 'aucs.csv', header=header)
    results_csv = CSVWriter(filename=Path(save_dir) / 'results.csv',
                            header=['epoch', 'final_epoch', 'final_loss', 'final_acc', 'final_auc', 'final_precision',
                                    'final_recall', 'final_f1_score'])

    best_model = copy.deepcopy({'state_dict': model.state_dict()})
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    model.train()
    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc, train_precision, train_recall, train_f1_score = \
            TRAIN[args.arch](args, epoch, train_set, model, criterion, optimizer, scheduler)
        valid_loss, valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score, *_ = \
            TEST[args.arch](args, valid_set, model, criterion, mode='Valid')
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score, *_ = \
            TEST[args.arch](args, test_set, model, criterion, mode='Test ')

        # Choose the best result
        if epoch >= args.warmup:
            if args.picked_method == 'acc':
                is_best = best_valid_acc.compare(valid_acc)
            elif args.picked_method == 'loss':
                is_best = best_valid_loss.compare(valid_loss)
            elif args.picked_method == 'auc':
                is_best = best_valid_auc.compare(valid_auc)
            elif args.picked_method == 'score':
                score = get_score(valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score)
                is_best = best_score.compare(score, epoch + 1, inplace=True)
            else:
                raise ValueError(f"picked_method error. ")
            if is_best:
                final_epoch = epoch + 1
                final_loss = test_loss
                final_acc = test_acc
                final_auc = test_auc
                final_precision = test_precision
                final_recall = test_recall
                final_f1_score = test_f1_score

            # Compute best result
            best_train_acc.compare(train_acc, epoch + 1, inplace=True)
            best_valid_acc.compare(valid_acc, epoch + 1, inplace=True)
            best_test_acc.compare(test_acc, epoch + 1, inplace=True)
            best_train_loss.compare(train_loss, epoch + 1, inplace=True)
            best_valid_loss.compare(valid_loss, epoch + 1, inplace=True)
            best_test_loss.compare(test_loss, epoch + 1, inplace=True)
            best_train_auc.compare(train_auc, epoch + 1, inplace=True)
            best_valid_auc.compare(valid_auc, epoch + 1, inplace=True)
            best_test_auc.compare(test_auc, epoch + 1, inplace=True)

            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'valid_acc': valid_acc,
                'best_valid_acc': best_valid_acc.best,
                'valid_auc': valid_auc,
                'best_valid_auc': best_valid_auc.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if is_best:
                best_model = copy.deepcopy(state)
                if args.save_model:
                    torch.save(state, Path(save_dir) / 'model_best.pth.tar')

            # Save
            losses_csv.write_row([epoch + 1, train_loss, valid_loss, test_loss,
                                  (best_train_loss.best, best_train_loss.epoch),
                                  (best_valid_loss.best, best_valid_loss.epoch),
                                  (best_test_loss.best, best_test_loss.epoch)])
            accs_csv.write_row([epoch + 1, train_acc, valid_acc, test_acc,
                                (best_train_acc.best, best_train_acc.epoch),
                                (best_valid_acc.best, best_valid_acc.epoch),
                                (best_test_acc.best, best_test_acc.epoch)])
            aucs_csv.write_row([epoch + 1, train_auc, valid_auc, test_auc,
                                (best_train_auc.best, best_train_auc.epoch),
                                (best_valid_auc.best, best_valid_auc.epoch),
                                (best_test_auc.best, best_test_auc.epoch)])
            results_csv.write_row(
                [epoch + 1, final_epoch, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score])

            print(
                f"Train acc: {train_acc:.4f}, Best: {best_train_acc.best:.4f}, Epoch: {best_train_acc.epoch:2}, "
                f"AUC: {train_auc:.4f}, Best: {best_train_auc.best:.4f}, Epoch: {best_train_auc.epoch:2}, "
                f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n"
                f"Valid acc: {valid_acc:.4f}, Best: {best_valid_acc.best:.4f}, Epoch: {best_valid_acc.epoch:2}, "
                f"AUC: {valid_auc:.4f}, Best: {best_valid_auc.best:.4f}, Epoch: {best_valid_auc.epoch:2}, "
                f"Loss: {valid_loss:.4f}, Best: {best_valid_loss.best:.4f}, Epoch: {best_valid_loss.epoch:2}\n"
                f"Test  acc: {test_acc:.4f}, Best: {best_test_acc.best:.4f}, Epoch: {best_test_acc.epoch:2}, "
                f"AUC: {test_auc:.4f}, Best: {best_test_auc.best:.4f}, Epoch: {best_test_auc.epoch:2}, "
                f"Loss: {test_loss:.4f}, Best: {best_test_loss.best:.4f}, Epoch: {best_test_loss.epoch:2}\n"
                f"Final Epoch: {final_epoch:2}, Final acc: {final_acc:.4f}, Final AUC: {final_auc:.4f}, Final Loss: {final_loss:.4f}\n"
            )

        # Early Stop
        if early_stop is not None:
            early_stop.update((best_valid_acc.best, best_valid_loss.best, best_valid_auc.best))
            if early_stop.is_stop():
                break

    return best_model


def test(args, test_set, model, criterion, mode='Test '):
    model.eval()
    with torch.no_grad():
        loss, acc, auc, precision, recall, f1_score, outputs_tensor, labels_tensor, case_id_list = \
            TEST[args.arch](args, test_set, model, criterion, mode)
        prob = torch.softmax(outputs_tensor, dim=1)
        _, pred = torch.max(prob, dim=1)
        preds = pd.DataFrame(columns=['label', 'pred', 'correct', *[f'prob{i}' for i in range(prob.shape[1])]])
        for i in range(len(case_id_list)):
            preds.loc[case_id_list[i]] = [
                labels_tensor[i].item(),
                pred[i].item(),
                labels_tensor[i].item() == pred[i].item(),
                *[prob[i][j].item() for j in range(prob.shape[1])],
            ]
        preds.index.rename('case_id', inplace=True)

    return loss, acc, auc, precision, recall, f1_score, preds


def run(args):
    # Configures
    init_seeds(args.seed)

    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep='_')  # increment run
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Dataset
    datasets, dim_patch = get_datasets(args)
    args.train_size, args.valid_size, args.test_size = \
        len(datasets['train']), len(datasets['valid']), len(datasets['test'])

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print("------------------- args ----------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-----------------------------------------------")

    # Model, Criterion, Optimizer and Scheduler
    model = create_model(args, dim_patch)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    # Start training
    model.zero_grad()
    best_model = train(args, datasets['train'], datasets['valid'], datasets['test'], model, criterion,
                       optimizer, scheduler, args.save_dir)
    print(f"Predicting...")
    model.load_state_dict(best_model['state_dict'])
    loss, acc, auc, precision, recall, f1_score, pred = test(args, datasets['test'], model, criterion, mode='Pred')
    score = get_score(acc, auc, precision, recall, f1_score)
    # Save results
    pred.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    final_res = pd.DataFrame(columns=['loss', 'acc', 'auc', 'precision', 'recall', 'f1_score', 'score'])
    final_res.loc[f'seed{args.seed}'] = [loss, acc, auc, precision, recall, f1_score, score]
    final_res.to_csv(str(Path(args.save_dir) / 'final_res.csv'))
    print(f'{final_res}\nPredicted Ending.\n')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='CAMELYON16_20x_s256',
                        help="Specify the data set used")
    parser.add_argument('--data_csv', type=str,
                        default='/path/to/data_csv',
                        help="")
    parser.add_argument('--data_split_json', type=str, default='/patch/to/data_split.json',
                        help="")
    parser.add_argument('--preload', action='store_true', default=False,
                        help="")
    parser.add_argument('--num_patches', type=int, default=None)
    parser.add_argument('--selecting', type=str, default='adaptive',
                        choices=['cluster_random', 'adaptive'])
    parser.add_argument('--fold_idx', type=int, default=None)
    # Train
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['AdamW', 'Adam', 'SGD', 'RMSprop'],
                        help="")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="")
    parser.add_argument('--epochs', type=int, default=50,
                        help="")
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="")
    parser.add_argument('--wdecay', default=1e-3, type=float,
                        help='')
    parser.add_argument('--picked_method', type=str, default='score',
                        help="loss, acc, auc or score")
    parser.add_argument('--patience', type=int, default=None,
                        help="")
    # Architecture
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--arch', default='CBMIL', type=str, choices=MODELS, help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--top_fuse', action='store_true', default=True)

    # Loss
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, choices=LOSSES,
                        help='loss name')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help="")
    # Save
    parser.add_argument('--base_save_dir', type=str, default='/dir/to/save/results')
    parser.add_argument('--save_dir', type=str, default=None, help="")
    parser.add_argument('--save_dir_flag', type=str, default=None)
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help="")
    parser.add_argument('--save_model', action='store_true', default=False, help="")
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=985,
                        help="")
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)

    # Global variables
    MODELS = ['CBMIL']

    LOSSES = ['CrossEntropyLoss']

    TRAIN = {'CBMIL': train_CBMIL}
    TEST = {'CBMIL': test_CBMIL}

    main()
