
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from data.randaug import *

def calculate_momentum_weight(momentum_loss, epoch):

    momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
    momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:])))/2+1/2).detach().cpu().numpy()

    return momentum_weight

class memoboosted_CIFAR100(CIFAR100):
    def __init__(self, sublist, args, **kwds):
        super().__init__(**kwds)
        self.txt = sublist
        self.args = args

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(100)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

        self.momentum_weight=np.empty(len(sublist))
        self.momentum_weight[:]=0

    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        memo_boosted_aug = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # RandAugment_prob(1, 30*self.momentum_weight[idx]*np.random.rand(1), 1.0*self.momentum_weight[idx]),
                RandAugment_prob(self.args.rand_k, self.args.rand_strength*self.momentum_weight[idx]*np.random.rand(1), 1.0*self.momentum_weight[idx]),
                transforms.ToTensor(),
            ])

        imgs = [memo_boosted_aug(img), memo_boosted_aug(img)]

        return torch.stack(imgs), idx
