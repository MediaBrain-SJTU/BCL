## Boosted-Contrastive-Learning

This repo provides a demo for the ICML 2022 paper "Contrastive Learning with Boosted Memorization" on the CIFAR-100-LT dataset. 

<div align="left">
  <img src="figures/methods.jpg" width="500px" />
</div>

## Quick Preview
A code snippet of the BCL is shown below. 

```python

train_datasets = memoboosted_CIFAR100(train_idx_list, args, root=args.data_folder, train=True)

# initialize momentum loss
shadow = torch.zeros(dataset_total_num).cuda()
momentum_loss = torch.zeros(args.epochs,dataset_total_num).cuda()

shadow, momentum_loss = train(train_loader, model, optimizer, scheduler, epoch, log, shadow, momentum_loss, args=args)
train_datasets.update_momentum_weight(momentum_loss, epoch)

```

During the training phase, track the momentum loss. 

```python

if epoch>1:
    new_average = (1.0 - args.momentum_loss_beta) * loss[batch_idx].clone().detach() + args.momentum_loss_beta * shadow[index[batch_idx]]
else:
    new_average = loss[batch_idx].clone().detach()
    
shadow[index[batch_idx]] = new_average
momentum_loss[epoch-1,index[batch_idx]] = new_average

```

## Implementation Details

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

### Content

- ```./bash_scripts```: bash scripts for running the code.
- ```./data```: datasets and augmentation.
- ```./models```: backbone models.
- ```./split```: imbalanced cifar-100 splits.
- ```eval_cifar.py```: code for linear probing evaluation.
- ```train.py```: code for training SimCLR and BCL.
- ```test.py```: code for testing SimCLR and BCL.
- ```utils.py```: utils (e.g. loss).

### Usage

**Train**

- SimCLR
```train SimCLR
bash bash_scripts/cifar-LT-SimCLR.sh
```

- BCL
```train BCL
bash bash bash_scripts/cifar-LT-BCL.sh
```

**Test**

```test
bash bash_scripts/cifar-LT-test.sh
```

### Extensions

**Steps to Implement Your Own Model**

- Add your model to ./models and load the model in train.py.
- Implement functions specfic to your models in train.py.

**Steps to Implement Other Datasets**

- Create long-tailed splits of the datasets and add to ./split.
- Implement the dataset (e.g. CIFAR100_index_bcl in data/cifar100.py).

