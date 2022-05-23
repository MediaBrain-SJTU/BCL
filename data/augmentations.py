import torchvision.transforms as transforms


cifar_tfs_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

cifar_tfs_test = transforms.Compose([
    transforms.ToTensor(),
])

  