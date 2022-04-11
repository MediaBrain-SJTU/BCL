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
from data.cifar100 import *
from data.augmentation import GaussianBlur
from eval_cifar import eval

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='', type=str, help='path to save checkpoint')
parser.add_argument('--data_folder', default='', type=str, help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset, [imagenet-LT, imagenet-100, places, cifar, cifar100]')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=500, type=int, help='print frequency')
parser.add_argument('--save_freq', default=500, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='if resume training')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--temperature', default=0.5, type=float, help='nt_xent temperature')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--trainSplit', type=str, default='trainIdxList.npy', help="train split")
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')
parser.add_argument('--strength', default=1.0, type=float, help='cifar augmentation, color jitter strength')
parser.add_argument('--resizeLower', default=0.1, type=float, help='resize smallest size')
parser.add_argument('--resizeHigher', default=1.0, type=float, help='resize largest size')
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--output_ch', default=128, type=int, help='proj head output feature number')
parser.add_argument('--eval_freq', default=20, type=int, help='eval frequency /epoch')
parser.add_argument('--test_10shot', action='store_true')
parser.add_argument('--test_50shot', action='store_true')
parser.add_argument('--test_100shot', action='store_true')

parser.add_argument('--momentum_loss_beta', type=float, default=0.9)
parser.add_argument('--bcl', action='store_true')



def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)
    
    rank = torch.distributed.get_rank()
    logName = "log.txt"

    log = logger(path=save_dir, local_rank=rank, log_name=logName)
    log.info(str(args))

    setup_seed(args.seed + rank)
    
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size


    if args.dataset == 'cifar100':
        num_class = 100
    else:
        assert False

    if args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=False, num_classes=num_class)
    else:
        assert False, "no such model"

    ch = model.fc.in_features
    from models.utils import proj_head
    model.fc = proj_head(ch, args.output_ch)
       
    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    cudnn.benchmark = True

    if args.dataset == "cifar100":
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * args.strength, 0.4 * args.strength,
                                                                          0.4 * args.strength, 0.1 * args.strength)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
      
        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.resizeLower, args.resizeHigher), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
        ])

        tfs_test = transforms.Compose([
              transforms.ToTensor(),
          ])
    else:
        assert False

    # dataset process
    if args.dataset == "cifar100":
        # the data distribution
        root = args.data_folder

        assert 'cifar100' in args.trainSplit
        train_idx = np.load('split/{}'.format(args.trainSplit))
        train_idx_list = list(train_idx)

        if args.bcl:
            train_datasets = CIFAR100_index_bcl(train_idx, root=root, train=True, transform=tfs_train, download=True)
        else:
            train_datasets = CIFAR100_index(train_idx_list, root=root, train=True, transform=tfs_train, download=True)
    else:
        assert False
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=False)

    # for eval
    eval_train_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=True, download=True, transform=tfs_test)

    eval_train_idx_fullshot = list(np.load('split/cifar100/cifar100_trainIdxList.npy'))
    eval_train_sampler_fullshot = SubsetRandomSampler(eval_train_idx_fullshot)
    eval_train_loader_fullshot = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=eval_train_sampler_fullshot, num_workers=2)

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

    eval_testset = torchvision.datasets.CIFAR100(root=args.data_folder, train=False, download=True, transform=tfs_test)
    eval_test_loader = torch.utils.data.DataLoader(eval_testset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=True)
    

    if args.dataset == "cifar100":
        if os.path.isdir(root):
            pass
        elif os.path.isdir(args.data_folder):
            root = args.data_folder

        val_train_datasets = datasets.CIFAR100(root=root, train=True, transform=tfs_test, download=True)
        val_train_sampler = SubsetRandomSampler(train_idx_list)
        val_train_loader = torch.utils.data.DataLoader(val_train_datasets, batch_size=batch_size, sampler=val_train_sampler)

        class_stat = [0 for _ in range(num_class)]
        for imgs, targets in val_train_loader:
            for target in targets:
                class_stat[target] += 1
        log.info("class distribution in training set is {}".format(class_stat))
        dataset_total_num = np.sum(class_stat)
        log.info("total sample number in training set is {}".format(dataset_total_num))


    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=0.2)
    elif args.scheduler == 'cosine':
        training_iters = args.epochs * len(train_loader)
        warm_up_iters = 10 * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    training_iters,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=warm_up_iters)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location="cuda")
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])

                for i in range((start_epoch - 1) * len(train_loader)):
                    scheduler.step()
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                log.info("cannot resume since lack of files")
                assert False

    if  args.bcl:
        shadow = torch.zeros(dataset_total_num).cuda()
        momentum_loss = torch.zeros(args.epochs,dataset_total_num).cuda()
        momentum_loss_ep, loss_ep = [], []

    for epoch in range(start_epoch, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)

        if args.bcl:
            shadow, momentum_loss, loss_index_ep, momentum_loss_index_ep, losses= train_bcl(train_loader, model, optimizer, scheduler, epoch, dataset_total_num, log, args.local_rank, rank, world_size, shadow, momentum_loss, args=args)
            momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
            momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:])))/2+1/2).detach().cpu().numpy()
            train_datasets.update_momentum_weight(momentum_weight)
        else:
            losses = train(train_loader, model, optimizer, scheduler, epoch, dataset_total_num, log, args.local_rank, rank, world_size, args=args)
            
        if (epoch+1) % args.eval_freq == 0 or epoch==0:

            acc_full = eval(eval_train_loader_fullshot, eval_test_loader, model, epoch, args=args)
            log.info("Accuracy fullshot {}".format(acc_full))

            if args.test_10shot:
                acc_few_1 = eval(eval_train_loader_10shot_1, eval_test_loader, model, epoch, args=args)
                acc_few_2 = eval(eval_train_loader_10shot_2, eval_test_loader, model, epoch, args=args)
                acc_few_3 = eval(eval_train_loader_10shot_3, eval_test_loader, model, epoch, args=args)
                acc_few_4 = eval(eval_train_loader_10shot_4, eval_test_loader, model, epoch, args=args)
                acc_few_5 = eval(eval_train_loader_10shot_5, eval_test_loader, model, epoch, args=args)
                acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
                log.info("Accuracy 10shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

            if args.test_50shot:
                acc_few_1 = eval(eval_train_loader_50shot_1, eval_test_loader, model, epoch, args=args)
                acc_few_2 = eval(eval_train_loader_50shot_2, eval_test_loader, model, epoch, args=args)
                acc_few_3 = eval(eval_train_loader_50shot_3, eval_test_loader, model, epoch, args=args)
                acc_few_4 = eval(eval_train_loader_50shot_4, eval_test_loader, model, epoch, args=args)
                acc_few_5 = eval(eval_train_loader_50shot_5, eval_test_loader, model, epoch, args=args)
                acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
                log.info("Accuracy 50shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

            if args.test_100shot:
                acc_few_1 = eval(eval_train_loader_100shot_1, eval_test_loader, model, epoch, args=args)
                acc_few_2 = eval(eval_train_loader_100shot_2, eval_test_loader, model, epoch, args=args)
                acc_few_3 = eval(eval_train_loader_100shot_3, eval_test_loader, model, epoch, args=args)
                acc_few_4 = eval(eval_train_loader_100shot_4, eval_test_loader, model, epoch, args=args)
                acc_few_5 = eval(eval_train_loader_100shot_5, eval_test_loader, model, epoch, args=args)
                acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
                log.info("Accuracy 100shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

        if rank == 0:
            save_model_freq = 2

            if epoch % save_model_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pt'))

            if epoch % args.save_freq == 0 and epoch > 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))
    
    if args.bcl:
        momentum_loss_tensor = momentum_loss.detach().cpu().numpy()
        np.save(os.path.join(save_dir, 'momentum_loss_{}'.format(epoch)),momentum_loss_tensor)

def train(train_loader, model, optimizer, scheduler, epoch, dataset_total_num, log, local_rank, rank, world_size, shadow=None, momentum_loss=None, args=None):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    for i, (inputs, label, index) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        batch_size = d[0]

        model.train()

        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)
        features = model(inputs)
        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)
        loss = nt_xent(features, t=args.temperature, average=False)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        losses.update(float(loss.mean().detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))
    
    return losses.avg


def train_bcl(train_loader, model, optimizer, scheduler, epoch, dataset_total_num, log, local_rank, rank, world_size, shadow=None, momentum_loss=None, args=None):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    label_index_ep = np.zeros(dataset_total_num)    
    loss_index_ep = np.zeros(dataset_total_num)
    momentum_loss_index_ep = np.zeros(dataset_total_num)
    momentum_weight_index_ep = np.zeros(dataset_total_num)

    for i, (inputs, label, index, momentum_weight) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        batch_size = d[0]

        model.train()

        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)
        features = model(inputs)
        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)
        loss = nt_xent(features, t=args.temperature, average=False)

        for count in range(batch_size):
            label_index_ep[index[count]]=label[count]
            loss_index_ep[index[count]]=loss[count].detach().cpu().numpy()
            momentum_weight_index_ep[index[count]]=momentum_weight[count].detach().cpu().numpy()
    
            if epoch>1:
                new_average = (1.0 - args.momentum_loss_beta) * loss[count].clone().detach() + args.momentum_loss_beta * shadow[index[count]]
            else:
                new_average = loss[count].clone().detach()
                
            shadow[index[count]] = new_average
            momentum_loss[epoch-1,index[count]] = new_average

            momentum_loss_index_ep[index[count]]=new_average.cpu().numpy() 

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        losses.update(float(loss.mean().detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))
        
    return shadow, momentum_loss, loss_index_ep, momentum_loss_index_ep, losses.avg

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()


