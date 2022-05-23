import argparse
import os
import torch
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import numpy as np
import torchvision
from data.cifar100 import *
from eval_cifar import eval
from models.simclr import SimCLR
from data.memoboosted_cifar100 import memoboosted_CIFAR100

parser = argparse.ArgumentParser(description='PyTorch Cifar100LT Self-supervised Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='', type=str, help='path to save checkpoint')
parser.add_argument('--data_folder', default='', type=str, help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar100', help="dataset-cifar100")
parser.add_argument('--trainSplit', type=str, default='trainIdxList.npy', help="train split")
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
parser.add_argument('--momentum_loss_beta', type=float, default=0.95)
parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength in randaugment(0-30)')


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
    
    # create log
    logName = "log.txt"
    log = logger(path=save_dir, log_name=logName)
    log.info(str(args))

    # model
    model = SimCLR(num_class=args.num_class, network=args.model).cuda()
 
    # data aug
    tfs_train, tfs_test = cifar_aug()

    # train loader
    train_idx = np.load('split/{}'.format(args.trainSplit))
    train_idx_list = list(train_idx)
    if args.bcl:
        train_datasets = memoboosted_CIFAR100(train_idx_list, args, root=args.data_folder, train=True)
    else:
        train_datasets = CIFAR100_index(train_idx_list, root=args.data_folder, train=True, transform=tfs_train, download=True)
    train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # dataset statistics
    class_stat = train_datasets.idxsNumPerClass
    log.info("class distribution in training set is {}".format(class_stat))
    dataset_total_num = np.sum(class_stat)
    log.info("total sample number in training set is {}".format(dataset_total_num))

    # eval loader
    eval_train_datasets = torchvision.datasets.CIFAR100(root=args.data_folder, train=True, download=True, transform=tfs_test)
    eval_train_loader_fullshot = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100/cifar100_trainIdxList.npy'))), num_workers=2)
    eval_train_loader_100shot_1 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split1_test_100shot.npy'))), num_workers=2)
    eval_train_loader_100shot_2 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split2_test_100shot.npy'))), num_workers=2)
    eval_train_loader_100shot_3 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split3_test_100shot.npy'))), num_workers=2)
    eval_train_loader_100shot_4 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split4_test_100shot.npy'))), num_workers=2)
    eval_train_loader_100shot_5 = torch.utils.data.DataLoader(eval_train_datasets,batch_size=1000, sampler=SubsetRandomSampler(list(np.load('split/cifar100_imbSub_with_subsets/cifar100_split5_test_100shot.npy'))), num_workers=2)

    eval_testset = torchvision.datasets.CIFAR100(root=args.data_folder, train=False, download=True, transform=tfs_test)
    eval_test_loader = torch.utils.data.DataLoader(eval_testset, batch_size=1000, shuffle=False, num_workers=1, pin_memory=True)
    
    # setting training schedule
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6 / args.lr, warmup_steps=10 * len(train_loader)))
            
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

        shadow, momentum_loss = train(train_loader, model, optimizer, scheduler, epoch, log, shadow, momentum_loss, args=args)
        if args.bcl:
            train_datasets.update_momentum_weight(momentum_loss, epoch)
     
        if (epoch+1) % args.eval_freq == 0 or epoch==0:
            # linear probing on full dataset 
            acc_full = eval(eval_train_loader_fullshot, eval_test_loader, model, epoch, args=args)
            log.info("Accuracy fullshot {}".format(acc_full))
            # linear probing on 100-shot dataset
            acc_few_1 = eval(eval_train_loader_100shot_1, eval_test_loader, model, epoch, args=args)
            acc_few_2 = eval(eval_train_loader_100shot_2, eval_test_loader, model, epoch, args=args)
            acc_few_3 = eval(eval_train_loader_100shot_3, eval_test_loader, model, epoch, args=args)
            acc_few_4 = eval(eval_train_loader_100shot_4, eval_test_loader, model, epoch, args=args)
            acc_few_5 = eval(eval_train_loader_100shot_5, eval_test_loader, model, epoch, args=args)
            acc_average = (acc_few_1[1]+acc_few_2[1]+acc_few_3[1]+acc_few_4[1]+acc_few_5[1])/5
            log.info("Accuracy 100shot {},{},{},{},{}, Average {}".format(acc_few_1, acc_few_2, acc_few_3, acc_few_4, acc_few_5, acc_average))

        if epoch % 2 == 0:
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),}, filename=os.path.join(save_dir, 'model.pt'))
        if epoch % args.save_freq == 0 and epoch > 0:
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),}, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))
    

def train(train_loader, model, optimizer, scheduler, epoch, log, shadow=None, momentum_loss=None, args=None):
    losses, data_time_meter, train_time_meter = AverageMeter(), AverageMeter(), AverageMeter()
    losses.reset()
    end = time.time()

    for i, (inputs, index) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()
        model.train()

        d = inputs.size()
        batch_size = d[0]
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)

        features = model(inputs)
        loss = nt_xent(features, t=args.temperature, average=False)

        for count in range(batch_size):
            if epoch>1:
                new_average = (1.0 - args.momentum_loss_beta) * loss[count].clone().detach() + args.momentum_loss_beta * shadow[index[count]]
            else:
                new_average = loss[count].clone().detach()
                
            shadow[index[count]] = new_average
            momentum_loss[epoch-1,index[count]] = new_average

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
        
    return shadow, momentum_loss

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()


