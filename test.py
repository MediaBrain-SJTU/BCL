import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet import resnet18
from utils import *
import torchvision.transforms as transforms
import torch.distributed as dist
import numpy as np
import torchvision
from data.cifar100 import CustomCIFAR100, CIFAR100_index
from data.augmentation import GaussianBlur
from eval_cifar import eval


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='', type=str, help='path to save checkpoint')
parser.add_argument('--data_folder', default='', type=str, help='dataset path')
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset, [imagenet-LT, imagenet-100, places, cifar, cifar100]')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
parser.add_argument('--save_freq', default=100, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='if resume training')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--temperature', default=0.5, type=float, help='nt_xent temperature')
parser.add_argument('--imagenetCustomSplit', type=str, default='', help="imagenet custom split")

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')
parser.add_argument('--strength', default=1.0, type=float, help='cifar augmentation, color jitter strength')
parser.add_argument('--resizeLower', default=0.1, type=float, help='resize smallest size')

# model setting
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--output_ch', default=128, type=int, help='proj head output feature number')
parser.add_argument('--eval_freq', default=20, type=int, help='eval frequency /epoch')

parser.add_argument('--test', action='store_true')
parser.add_argument('--test_fullshot', action='store_true')
parser.add_argument('--test_10shot', action='store_true')
parser.add_argument('--test_50shot', action='store_true')
parser.add_argument('--test_100shot', action='store_true')



def main():
    global args
    args = parser.parse_args()

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)
    
    rank = torch.distributed.get_rank()

    setup_seed(args.seed + rank)
    
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    num_class = 100

    if args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=False, num_classes=num_class)
    
    ch = model.fc.in_features

    from models.utils import proj_head
    model.fc = proj_head(ch, args.output_ch)

    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    cudnn.benchmark = True

    tfs_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    

    # for eval
    if args.dataset == "cifar100":
        eval_train_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=True, download=True, transform=tfs_test)
        eval_train_idx_fullshot = list(np.load('split/cifar100/cifar100_trainIdxList.npy'))
        eval_train_sampler_fullshot = SubsetRandomSampler(eval_train_idx_fullshot)
        eval_train_loader_fullshot = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=eval_train_sampler_fullshot, num_workers=2)

        eval_testset = torchvision.datasets.CIFAR100(root=args.data_folder, train=False, download=True, transform=tfs_test)
        eval_test_loader = torch.utils.data.DataLoader(eval_testset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=True)

        if args.test_10shot:
            eval_train_loader_10shot_1 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split1_test_10shot.npy'))), num_workers=2)
            eval_train_loader_10shot_2 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split2_test_10shot.npy'))), num_workers=2)
            eval_train_loader_10shot_3 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split3_test_10shot.npy'))), num_workers=2)
            eval_train_loader_10shot_4 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split4_test_10shot.npy'))), num_workers=2)
            eval_train_loader_10shot_5 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split5_test_10shot.npy'))), num_workers=2)
        if args.test_50shot:
            eval_train_loader_50shot_1 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split1_test_50shot.npy'))), num_workers=2)
            eval_train_loader_50shot_2 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split2_test_50shot.npy'))), num_workers=2)
            eval_train_loader_50shot_3 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split3_test_50shot.npy'))), num_workers=2)
            eval_train_loader_50shot_4 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split4_test_50shot.npy'))), num_workers=2)
            eval_train_loader_50shot_5 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split5_test_50shot.npy'))), num_workers=2)
        if args.test_100shot:
            eval_train_loader_100shot_1 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split1_test_100shot.npy'))), num_workers=2)
            eval_train_loader_100shot_2 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split2_test_100shot.npy'))), num_workers=2)
            eval_train_loader_100shot_3 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split3_test_100shot.npy'))), num_workers=2)
            eval_train_loader_100shot_4 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split4_test_100shot.npy'))), num_workers=2)
            eval_train_loader_100shot_5 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split5_test_100shot.npy'))), num_workers=2)


    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)        

    epoch=0
    if args.test and args.dataset=='cifar100':
        if args.test_fullshot:
            acc_full = eval(eval_train_loader_fullshot, eval_test_loader, model, epoch, args=args)
            print("Accuracy fullshot {}".format(acc_full))

        if args.test_10shot:
            acc_few_1 = eval(eval_train_loader_10shot_1, eval_test_loader, model, epoch, args=args)
            acc_few_2 = eval(eval_train_loader_10shot_2, eval_test_loader, model, epoch, args=args)
            acc_few_3 = eval(eval_train_loader_10shot_3, eval_test_loader, model, epoch, args=args)
            acc_few_4 = eval(eval_train_loader_10shot_4, eval_test_loader, model, epoch, args=args)
            acc_few_5 = eval(eval_train_loader_10shot_5, eval_test_loader, model, epoch, args=args)

            acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
            print("Accuracy 10shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

        if args.test_50shot:
            acc_few_1 = eval(eval_train_loader_50shot_1, eval_test_loader, model, epoch, args=args)
            acc_few_2 = eval(eval_train_loader_50shot_2, eval_test_loader, model, epoch, args=args)
            acc_few_3 = eval(eval_train_loader_50shot_3, eval_test_loader, model, epoch, args=args)
            acc_few_4 = eval(eval_train_loader_50shot_4, eval_test_loader, model, epoch, args=args)
            acc_few_5 = eval(eval_train_loader_50shot_5, eval_test_loader, model, epoch, args=args)

            acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
            print("Accuracy 50shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))
        
        if args.test_100shot:
            acc_few_1 = eval(eval_train_loader_100shot_1, eval_test_loader, model, epoch, args=args)
            acc_few_2 = eval(eval_train_loader_100shot_2, eval_test_loader, model, epoch, args=args)
            acc_few_3 = eval(eval_train_loader_100shot_3, eval_test_loader, model, epoch, args=args)
            acc_few_4 = eval(eval_train_loader_100shot_4, eval_test_loader, model, epoch, args=args)
            acc_few_5 = eval(eval_train_loader_100shot_5, eval_test_loader, model, epoch, args=args)

            acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
            print("Accuracy 100shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

        return
        


if __name__ == '__main__':
    main()


