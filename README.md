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

## Usage
First, clone the repository locally:
```
git clone https://github.com/Lee-JH-KR/Continual-datasets.git
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
It can be used in various scenarios by changing `--dataset` and `--num_tasks` as shown below:

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


Also available in Slurm by changing options on `train.sh`
