import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as transforms
from torchvision import datasets


def create_long_tail_distribution(dataset, imbalance_factor):

    targets = np.array(dataset.targets)
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    max_samples = max(class_counts.values())
    class_indices = []
    for cls in range(num_classes):
        num_samples = int(max_samples * (imbalance_factor ** (cls / (num_classes - 1.0))))
        indices = np.where(targets == cls)[0]
        np.random.shuffle(indices)
        class_indices.extend(indices[:num_samples])
    return class_indices


def create_balanced_validation_set(val_subset, train_dataset):

    val_indices = np.array(val_subset.indices)
    val_labels = np.array([train_dataset.targets[i] for i in val_indices])
    _, val_indices_bal = train_test_split(val_indices, test_size=0.5, stratify=val_labels)
    return Subset(train_dataset, val_indices_bal)



def load_datasets(ID_dataset, batch_size, val_size=0.05, imbalance_factor=0.0, noise=False):
 
    name = ID_dataset.upper()
    if name in ["MNIST", "DMNIST"]:
        return dataloaders_mnist(batch_size, val_size, imbalance_factor, noise)
    elif name == "CIFAR-10":
        return dataloaders_cifar10(batch_size, val_size, imbalance_factor, noise)
    elif name == "CIFAR-100":
        return dataloaders_cifar100(batch_size, val_size, imbalance_factor, noise)
    else:
        raise ValueError(f"Unsupported dataset: {ID_dataset}")

def dataloaders_mnist(batch_size, val_size, imbalance_factor=0.0, noise=False, ambig_ratio=0.5):

    root = "./data"
    num_workers = 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # OOD datasets (always loaded)
    fmnist_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    kmnist_dataset = datasets.KMNIST(root=root, train=False, download=True, transform=transform)

    if not noise:
        base_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        base_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        if imbalance_factor == 0:
            train_size = int((1 - val_size) * len(base_train))
            val_size_actual = len(base_train) - train_size
            train_subset, val_subset = random_split(base_train, [train_size, val_size_actual])
            val_subset = create_balanced_validation_set(val_subset, base_train)
        else:
            long_tail_indices = create_long_tail_distribution(base_train, imbalance_factor)
            lt_dataset = Subset(base_train, long_tail_indices)
            train_size = int((1 - val_size) * len(lt_dataset))
            val_size_actual = len(lt_dataset) - train_size
            train_subset, val_subset = random_split(lt_dataset, [train_size, val_size_actual])
            val_subset = create_balanced_validation_set(val_subset, base_train)

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        from ddu_dirty_mnist import DirtyMNIST
        dirty_train = DirtyMNIST(root=root, train=True, download=True)
        dirty_test = DirtyMNIST(root=root, train=False, download=True)

        if imbalance_factor > 0:
            print(f"[INFO] Applying imbalance_factor={imbalance_factor} to DirtyMNIST")
            if isinstance(dirty_train, torch.utils.data.ConcatDataset):
                all_targets = []
                for sub in dirty_train.datasets:
                    if hasattr(sub, "targets"):
                        all_targets.extend(sub.targets)
                    elif hasattr(sub, "labels"):
                        all_targets.extend(sub.labels)
                dirty_train.targets = all_targets

            long_tail_indices = create_long_tail_distribution(dirty_train, imbalance_factor)
            lt_dataset = Subset(dirty_train, long_tail_indices)
        else:
            lt_dataset = dirty_train

        total_len = len(lt_dataset)
        clean_size = int(total_len * (1 - ambig_ratio))
        ambig_size = total_len - clean_size

        clean_indices = np.arange(0, clean_size)
        ambig_indices = np.arange(clean_size, total_len)
        combined_indices = np.concatenate([clean_indices, ambig_indices])
        np.random.shuffle(combined_indices)
        lt_dataset = Subset(lt_dataset, combined_indices)

        train_size = int((1 - val_size) * len(lt_dataset))
        val_size_actual = len(lt_dataset) - train_size
        train_subset, val_subset = random_split(lt_dataset, [train_size, val_size_actual])


        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader = DataLoader(dirty_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    fmnist_loader = DataLoader(fmnist_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kmnist_loader = DataLoader(kmnist_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader, testloader, fmnist_loader, kmnist_loader



def dataloaders_cifar10(batch_size, val_size, imbalance_factor=0.0, noise=False):
    root = "./data"
    num_workers = 4
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    base_valid = datasets.CIFAR10(root=root, train=True, download=True, transform=test_transform)
    base_test = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    if imbalance_factor == 0:
        num_train = len(base_train)
        indices = np.random.permutation(num_train)
        split = int(np.floor(val_size * num_train))
        train_indices, val_indices = indices[split:], indices[:split]
        train_subset = Subset(base_train, train_indices)
        val_subset = Subset(base_valid, val_indices)
    else:
        long_tail_indices = create_long_tail_distribution(base_train, imbalance_factor)
        lt_dataset = Subset(base_train, long_tail_indices)
        train_size = int((1 - val_size) * len(lt_dataset))
        val_size_actual = len(lt_dataset) - train_size
        train_subset, val_subset = random_split(lt_dataset, [train_size, val_size_actual])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    svhn_dataset = datasets.SVHN(root=root, split="test", download=True, transform=test_transform)
    cifar100_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader, svhn_loader, cifar100_loader

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def dataloaders_cifar100(batch_size, val_size, imbalance_factor=0.0, noise=False):
    root = "./data"
    num_workers = 4 # Increased slightly for better throughput
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean, std)

    # 1. Define Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(), 
        normalize
    ])

    ood_transform = transforms.Compose([
        transforms.Resize(32),  
        transforms.ToTensor(),
        normalize
    ])

    base_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
    base_targets = np.array(base_dataset.targets)

    if imbalance_factor and imbalance_factor > 0:
        # Assuming create_long_tail_distribution is defined in your scope
        lt_indices = np.array(create_long_tail_distribution(base_dataset, imbalance_factor))
        indices_pool, labels_pool = lt_indices, base_targets[lt_indices]
    else:
        indices_pool, labels_pool = np.arange(len(base_dataset)), base_targets

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    split_train_idx, split_val_idx = next(sss.split(indices_pool, labels_pool))
    
    train_indices = indices_pool[split_train_idx]
    val_indices = indices_pool[split_val_idx]

    train_subset = TransformedSubset(Subset(base_dataset, train_indices.tolist()), transform=train_transform)
    val_subset = TransformedSubset(Subset(base_dataset, val_indices.tolist()), transform=eval_transform)
    
    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)

    svhn_dataset = datasets.SVHN(root=root, split="test", download=True, transform=eval_transform)
    tin_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/test", transform=ood_transform)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tin_loader = DataLoader(tin_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader, svhn_loader, tin_loader

