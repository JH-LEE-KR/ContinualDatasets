# Continual Learning Datasets

This repository contains frequently used CL datasets and simple train and evaluation code.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Datasets
The available datasets are as follows:
### Task & Class Incremental
- CIFAR-100
- CIFAR-10
- MNIST
- FashionMNIST
- SVHN
- NotMNIST
- Flower102
- Stanford Cars196
- CUB200
- Indoor Scene67
- TinyImagenet
- Imagenet-R
### Domain Incremental
- CORe50
- DomainNet

## Usage
First, clone the repository locally:
```
git clone https://github.com/JH-LEE-KR/Continual-datasets.git
cd Continual-datasets
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Training
It can be used in various scenarios by changing `--dataset` and `--num_tasks` as shown below (default: class incremental):

**Split-CIFAR100 with 10 tasks**
```
python main.py --dataset Split-CIFAR100 --num_tasks 10
```


**Split-Imagenet-R with 10 tasks**
```
python main.py --dataset Split-Imagenet-R --num_tasks 10
```


**5 datasets, 5 tasks with MNIST, Fashion-MNIST, NotMNIST, CIFAR10, SVHN**
```
python main.py --dataset 5-datasets --num_tasks 5
```


**Sequence of datasets (CUB200,TinyImagenet,Scene67,Cars196,Flower102) with 5 tasks**
```
python main.py --dataset CUB200,TinyImagenet,Scene67,Cars196,Flower102 --num_tasks 5
```


You can customize scenario, as below:
**Split-TinyImagenet with 10 tasks**
```
python main.py --dataset Split-TinyImagenet --num_tasks 10
```


**Split-CUB200 with 10 tasks**
```
python main.py --dataset Split-CUB200 --num_tasks 10
```


**Sequence of datasets (CIFAR100,CUB200,TinyImagenet,Scene67,Cars196,Flower102,Imagenet-R) with 7 tasks**
```
python main.py --dataset IFAR100,CUB200,TinyImagenet,Scene67,Cars196,Flower102,Imagenet-R --num_tasks 7
```


**Domain Incremental CORe50 with 7 tasks**
```
python main.py --dataset CORe50 --num_tasks 7 --domain_inc --no_train_mask
```


**Domain Incremental DomainNet with 6 tasks**
```
python main.py --dataset DomainNet --num_tasks 6 --domain_inc --no_train_mask
```

**Options**
```
--train_mask, if using the class mask at training.
--no_train_mask, if domain incremental setting, not using the class mask at training.
--task_inc, if doing task incremental.
--domain_inc, if doing domain incremental.
--shuffle, shuffle the data order.
```

Also available in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> by changing options on `train.sh`
