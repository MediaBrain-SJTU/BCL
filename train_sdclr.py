import argparse
import os
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from eval_cifar import eval
from data.cifar100 import CIFAR100_index
from data.memoboosted_cifar100 import memoboosted_CIFAR100
from data.augmentations import cifar_tfs_train, cifar_tfs_test
from models.sdclr import SDCLR, Mask
from losses.nt_xent import NT_Xent_Loss

parser = argparse.ArgumentParser(description='PyTorch Cifar100-LT Self-supervised Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('--data_folder', default='', type=str, help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar100', help="dataset-cifar100")
parser.add_argument('--trainSplit', type=str, default='', help="train split")
parser.add_argument("--gpus", type=str, default="0", help="gpu id sequence split by comma")
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='num workers')
parser.add_argument('--model', default='resnet18', type=str, help='model type')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=2000, type=int, help='training epochs')
parser.add_argument('--num_class', default=100, type=int, help='num class')
parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
parser.add_argument('--save_freq', default=500, type=int, help='save frequency /epoch')
parser.add_argument('--eval_freq', default=20, type=int, help='eval frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='resume training')
parser.add_argument('--lr', default=0.5, type=float, help='optimizer lr')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--temperature', default=0.2, type=float, help='nt_xent temperature')
parser.add_argument('--bcl', action='store_true', help='boosted contrastive learning')
parser.add_argument('--momentum_loss_beta', type=float, default=0.97)
parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength in randaugment(0-30)')
parser.add_argument('--prune_percent', type=float, default=0.9, help="whole prune percentage")
parser.add_argument('--random_prune_percent', type=float, default=0, help="random prune percentage")


def main():
    global args
    args = parser.parse_args()

    # create dir
    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    # gpu 
    gpus = list(map(lambda x: torch.device('cuda', x), [int(e) for e in args.gpus.strip().split(",")]))
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)
    
    # init log
    log = logger(path=save_dir, log_name="log.txt")
    log.info(str(args))

    # create model
    model = SDCLR(num_class=args.num_class, network=args.model).cuda()

    # criterion
    criterion = NT_Xent_Loss(temp=args.temperature, average=False)

    # data augmentations
    tfs_train, tfs_test = cifar_tfs_train, cifar_tfs_test
    # loading data
    train_idx_list = list(np.load('split/{}'.format(args.trainSplit)))
    if args.bcl:
        train_datasets = memoboosted_CIFAR100(train_idx_list, args, root=args.data_folder, train=True)
    else:
        train_datasets = CIFAR100_index(train_idx_list, root=args.data_folder, train=True, transform=tfs_train, download=True)
    eval_train_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=True, download=True, transform=tfs_test)
    eval_test_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=False, download=True, transform=tfs_test)

    train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(eval_train_datasets, batch_size=1000, num_workers=args.num_workers, sampler=SubsetRandomSampler(list(np.load('split/cifar100/cifar100_trainIdxList.npy'))))
    eval_test_loader = torch.utils.data.DataLoader(eval_test_datasets, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # dataset statistics
    class_stat = train_datasets.idxsNumPerClass
    dataset_total_num = np.sum(class_stat)
    log.info("class distribution in training set is {}".format(class_stat))

    # optimizer, training schedule
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6 / args.lr, warmup_steps=10 * len(train_loader)))

    # optionally resume from a checkpoint 
    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location="cuda")
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            for i in range((start_epoch - 1) * len(train_loader)):
                scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            
    # initialize momentum loss
    shadow = torch.zeros(dataset_total_num).cuda()
    momentum_loss = torch.zeros(args.epochs,dataset_total_num).cuda()
    
    # training
    for epoch in range(1, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        shadow, momentum_loss = train_prune(train_loader, model, criterion, optimizer, scheduler, epoch, log, shadow, momentum_loss, args=args)
        if args.bcl:
            train_datasets.update_momentum_weight(momentum_loss, epoch)
     
        if (epoch+1) % args.eval_freq == 0 or epoch==0:
            # linear probing on full dataset 
            acc_full = eval(eval_train_loader, eval_test_loader, model, epoch, args=args)
            log.info("Accuracy fullshot {}".format(acc_full))

        if epoch % 2 == 0:
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),}, filename=os.path.join(save_dir, 'model.pt'))
        if epoch % args.save_freq == 0 and epoch > 0:
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),}, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))
    

def train_prune(train_loader, model, criterion, optimizer, scheduler, epoch, log, shadow=None, momentum_loss=None, args=None):

    pruneMask = Mask(model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent

    log.info("current prune percent is {}".format(prunePercent))
    if randomPrunePercent > 0:
        log.info("random prune percent is {}".format(randomPrunePercent))

    losses, data_time_meter, train_time_meter = AverageMeter(), AverageMeter(), AverageMeter()
    losses.reset()
    end = time.time()

    # prune every epoch
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

    for i, (inputs, index) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        inputs = inputs.cuda(non_blocking=True)
        inputs_1 = inputs[:, 0, ...]
        inputs_2 = inputs[:, 1, ...]

        model.train()
        optimizer.zero_grad()

        # calculate the grad for non-pruned network
        with torch.no_grad():
            # branch with pruned network
            model.backbone.set_prune_flag(True)
            features_2_noGrad = model(inputs_2).detach()
        model.backbone.set_prune_flag(False)
        features_1 = model(inputs_1)

        loss = criterion(features_1, features_=features_2_noGrad)

        for count in range(inputs.size()[0]):
            if epoch>1:
                new_average = (1.0 - args.momentum_loss_beta) * loss[count].clone().detach() + args.momentum_loss_beta * shadow[index[count]]
            else:
                new_average = loss[count].clone().detach()
                
            shadow[index[count]] = new_average
            momentum_loss[epoch-1,index[count]] = new_average

        loss.mean().backward()
        losses.update(float(loss.mean().detach().cpu()), inputs.shape[0])

        # calculate the grad for pruned network
        features_1_no_grad = features_1.detach()
        model.backbone.set_prune_flag(True)
        features_2 = model(inputs_2)

        loss = criterion(features_1_no_grad, features_=features_2)
        loss.mean().backward()

        optimizer.step()

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
        
    return shadow, momentum_loss


if __name__ == '__main__':
    main()


