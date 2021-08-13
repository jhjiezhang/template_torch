import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.resnet import resnet18, resnet50


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss

def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "loss={:.4f} acc={:.2f}%"
    train_loader.sampler.set_epoch(epoch)
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            optimizer.zero_grad()
            epochs.set_description(description.format(avg_loss, acc))
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(train_loader.dataset) * 100


def get_cifar100():
    data_dir = "/youtu-face-identify-public/jiezhang/data/"
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = datasets.CIFAR100(data_dir, train=True, download=True,
                                  transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=False,
                                               num_workers=4, sampler=train_sampler)
    test_set = datasets.CIFAR100(data_dir, train=False, transform=transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256, shuffle=False, num_workers=4, sampler=test_sampler)

    return train_loader, test_loader


parser = argparse.ArgumentParser()

# DDP：DDP backend初始化
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
train_loader, test_loader = get_cifar100()

# 构造模型
model = resnet18(num_classes=100).cuda()

model = DDP(model)
MILESTONES = [60, 120, 160]
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES,
                                                 gamma=0.2)  # learning rate decay
# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().cuda()

for i in tqdm(range(200)):
    train(model, train_loader, optimizer, i)
    acc, test_loss = test(model, test_loader)
    torch.save(model.state_dict(), "tmp.pkl")
# model.load_state_dict(torch.load("tmp.pkl"))
# acc, test_loss = test(model, test_loader)
# python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 29501 distr_cifar100.py
