import time
import torch
from torch import nn
from utils import AverageMeter
import numpy as np
import torch.optim as optim


def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500, batch_size=1000):
    
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)

    clf.cuda()

    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
    
        perm = torch.randperm(len(x_train))
        n_batch = int(np.ceil(len(x_train)*1.0/batch_size))
        for ii in range(n_batch):
            optimizer.zero_grad()
            mask = perm[ii*batch_size:(ii+1)*batch_size]
            criterion(clf(x_train[mask]), y_train[mask]).backward()
            optimizer.step()
        
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }

    del clf
    return acc

def eval(train_loader, test_loader, model, epoch, args=None):
    
    model.eval()
    
    fc = model.module.fc
    model.module.fc = nn.Identity()

    with torch.no_grad():

        model.eval()
        x_train = []
        y_train = []

        x_test = []
        y_test = []

        for i, (inputs, labels) in enumerate(train_loader):
        
            inputs = inputs.cuda()
            features = model(inputs)

            x_train.append(features.detach())
            y_train.append(labels.detach())

        for i, (inputs, labels) in enumerate(test_loader):
            
            inputs = inputs.cuda()
            features = model(inputs)

            x_test.append(features.detach())
            y_test.append(labels.detach())

        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0).cuda()

        x_test = torch.cat(x_test, dim=0)
        y_test = torch.cat(y_test, dim=0).cuda()

    acc = eval_sgd(x_train, y_train, x_test, y_test)

    model.module.fc = fc

    return acc