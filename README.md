## Boosted-Contrastive-Learning

## Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

## Content

- ```./bash_scripts```: bash scripts for running the code.
- ```./data```: datasets and augmentation.
- ```./models```: backbone models.
- ```./split```: imbalanced cifar-100 splits.
- ```eval_cifar.py```: code for linear probing evaluation.
- ```train.py```: code for training SimCLR and BCL.
- ```test.py```: code for testing SimCLR and BCL.
- ```utils.py```: utils(e.g. loss).

## Usage

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

## Extensions

**Steps to Implement Your Own Model**

- Add model to ./models and load model in train.py.
- Implement functions specfic to your models in train.py.

**Steps to Implement Other Datasets**

- Create long-tailed splits of the datasets and add to ./split.
- Implement the dataset(e.g. CIFAR100_index_bcl in data/cifar100.py).

## Reference Code

[1] SDCLR: https://github.com/VITA-Group/SDCLR

[2] RandAugment: https://github.com/ildoonet/pytorch-randaugment

[3] Linear probing setting in W-MSE: https://github.com/htdt/self-supervised
