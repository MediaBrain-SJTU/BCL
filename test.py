import argparse
import torch
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import torchvision.transforms as transforms
import numpy as np
import torchvision
from eval_cifar import eval
from models.simclr import SimCLR
from models.sdclr import SDCLR, Mask

parser = argparse.ArgumentParser(description='PyTorch Cifar100-LT Testing')
parser.add_argument('--save_dir', default='', type=str, help='path to save checkpoint')
parser.add_argument('--data_folder', default='', type=str, help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar100', help="dataset-cifar100")
parser.add_argument("--gpus", type=str, default="0", help="gpu id sequence split by comma")
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--model', default='resnet18', type=str, help='model type')
parser.add_argument('--test_fullshot', action='store_true')
parser.add_argument('--test_100shot', action='store_true')
parser.add_argument('--test_50shot', action='store_true')
parser.add_argument('--prune', action='store_true')
parser.add_argument('--prune_percent', type=float, default=0, help="whole prune percentage")


def main():
    global args
    args = parser.parse_args()

    # gpu 
    gpus = list(map(lambda x: torch.device('cuda', x), [int(e) for e in args.gpus.strip().split(",")]))
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)

    if not args.prune:
        model = SimCLR(num_class=100, network=args.model).cuda()
    else:
        model = SDCLR(num_class=100, network=args.model).cuda()
        args.prune_percent = 0.9

    tfs_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # for eval
    eval_train_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=True, download=True, transform=tfs_test)
    eval_train_idx_fullshot = list(np.load('split/cifar100/cifar100_trainIdxList.npy'))
    eval_train_sampler_fullshot = SubsetRandomSampler(eval_train_idx_fullshot)
    eval_train_loader_fullshot = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=eval_train_sampler_fullshot, num_workers=2)

    eval_testset = torchvision.datasets.CIFAR100(root=args.data_folder, train=False, download=True, transform=tfs_test)
    eval_test_loader = torch.utils.data.DataLoader(eval_testset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=True)

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

    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)        

    epoch=0
    if args.test_fullshot:
        acc_full = eval(eval_train_loader_fullshot, eval_test_loader, model, epoch, args=args)
        print("Accuracy fullshot {}".format(acc_full))

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


