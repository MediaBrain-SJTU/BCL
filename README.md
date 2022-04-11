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

To train BCL, please run this command:
```train BCL
bash bash bash_scripts/cifar-LT-BCL.sh
```
To train SimCLR, please run this command:
```train SimCLR
bash bash_scripts/cifar-LT-SimCLR.sh
```

**Test**
To test BCL or SimCLR, please run this command:
```test
bash bash_scripts/cifar-LT-test.sh
```

## Reference Code

[1] SDCLR: https://github.com/VITA-Group/SDCLR

[2] RandAugment: https://github.com/ildoonet/pytorch-randaugment

[3] Linear probing setting in W-MSE: https://github.com/htdt/self-supervised
