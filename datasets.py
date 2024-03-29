# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from continual_datasets.continual_datasets import *

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset == '5-datasets':
        dataset_list = ['CIFAR10', 'MNIST', 'FashionMNIST', 'SVHN', 'NotMNIST']

    elif args.dataset == 'iDigits':
        dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']

    else:
        dataset_list = args.dataset.split(',')

    splited_dataset = list()
    args.nb_classes = 0

    for i, dataset in enumerate(dataset_list):
        dataset_train, dataset_val = get_dataset(
            dataset=dataset if not dataset.startswith('Split-') else dataset.replace('Split-',''),
            transform_train=transform_train,
            transform_val=transform_val,
            args=args,
        )

        if args.dataset.startswith('Split-') and len(dataset_list) == 1:
            args.nb_classes = len(dataset_train.classes)

            splited_dataset, mask = split_single_dataset(dataset_train, dataset_val, args)

            splited_dataset.append((dataset_train, dataset_val))

            if class_mask is not None:
                class_mask = mask
        else:
            if not args.domain_inc:
                transform_target = Lambda(target_transform, args.nb_classes)
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
            
            if args.domain_inc:
                if args.dataset == 'CORe50':
                    splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                elif args.dataset == 'iDigits':
                    splited_dataset.append((dataset_train, dataset_val))
                else:
                    splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_val))]
            else:
                splited_dataset.append((dataset_train, dataset_val))
            
            # DIL setting should not use class mask, because of assumption the number of classes is the same for all tasks
            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])

        if args.domain_inc and args.dataset in ['CORe50', 'DomainNet', 'PermutedMNIST', 'iDigits']:
            # DIL setting assume that the number of classes are the same for all tasks
            # So, we should not add the number of classes for each task
            # Just add one for val set of the first task
            args.nb_classes = len(splited_dataset[0][1].classes)
        elif not args.dataset.startswith('Split-'):
            args.nb_classes += len(splited_dataset[i][1].classes)
        else:
            pass

    if args.shuffle:
        if class_mask is not None:
            zipped = list(zip(splited_dataset, class_mask))
            random.shuffle(zipped)
            splited_dataset, class_mask = zip(*zipped)
        else:
            random.shuffle(splited_dataset)

    for i in range(args.num_tasks):
        dataset_train, dataset_val = splited_dataset[i]

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'PermutedMNIST':
        dataset_train = [PermutedMNIST(args.data_path, train=True, download=True, transform=transform_train, random_seed=i) for i in range(args.num_tasks)]
        dataset_val = [PermutedMNIST(args.data_path, train=False, download=True, transform=transform_val, random_seed=i) for i in range(args.num_tasks)]

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.label_shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)