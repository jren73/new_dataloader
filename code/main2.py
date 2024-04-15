import torch
import torchvision
from torchvision.models import resnet18
import ultraimport
mydl = ultraimport('util.py', 'MyCustomDataLoader')

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm

import argparse
import time

from resnet import *

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default='sgd', choices=['sgd', 'sgd_hardware'], help='')
parser.add_argument('--micro_batch_size', type=int, default=16, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--repeat_times', type=int, default=1, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--lr', type=float, default=0.05, help='')
parser.add_argument('--weight_decay', type=float, default=0, help='')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenetv2'], help='')
args = parser.parse_args()


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms for data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                          shuffle=True, num_workers=2)

trainloader = mydl(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                         shuffle=False, num_workers=2)

testloader = mydl(testset, batch_size=100,
                                         shuffle=False, num_workers=1)

if args.model == 'resnet18':
    # model = resnet18(pretrained=False, num_classes=10).to(device)
    model = ResNet18().to(device)
elif args.model == 'mobilenetv2':
    model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training loop
cur_step = 0

run_time = []
err = []
test_acc = []

#create result folder if not exist
import os
if not os.path.exists("results"):
    os.makedirs("results")

prefix = "results/" + args.model + "_" + args.alg + "_" + str(args.micro_batch_size) + "_" + str(args.batch_size) + "_" \
                + str(args.repeat_times) + "_" + str(args.epochs) + "_" + str(args.lr) + "_" + str(args.weight_decay) + "_"

start_time = time.time()

test_time = 0

cur_epoch = 0
# while cur_step < args.total_steps:
for epoch in range(cur_epoch, args.epochs):
    print("cur_epoch", epoch, "start a new epoch!!!")
    # scheduler.step()

    # if epoch == 0:
    #     optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
    # elif epoch == 150:
    #     optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay = 4e-5, momentum = 0.9)
    # elif epoch == 225:
    #     optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 4e-5, momentum = 0.9)

    #do test every epoch

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        permute = torch.randperm(inputs.size(0))
        if inputs.size(0) < args.batch_size:
            print("skip!!!", cur_step)
            continue
        for j in range(args.repeat_times):
                      

            optimizer.zero_grad()

            start = (j*args.micro_batch_size) % args.batch_size
            if start == args.batch_size:
                start = 0
                permute = torch.randperm(inputs.size(0))

            index_range = permute[slice(start, start+args.micro_batch_size)]

            outputs = model(inputs[index_range])
            loss = criterion(outputs, labels[index_range])
            loss.backward()
            err.append(loss.item())
            if np.isnan(loss.item()): import pdb; pdb.set_trace()
            optimizer.step()

            cur_step += 1
            end_time = time.time()
            run_time.append(end_time - start_time - test_time)

            #print and save results as npy file every 100 steps
            if cur_step % 20 == 0:

                print(f"alg: {args.alg}, Steps: {cur_step}, Time: {round(end_time - start_time - test_time, 2)}, Loss: {round(loss.item(),2)}")

                np.save(prefix + "run_time.npy", run_time)
                np.save(prefix + "err.npy", err)
                np.save(prefix + "test_acc.npy", test_acc)

            if cur_step % 200 == 0:
                test_start = time.time()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data_test in testloader:
                        images_test, labels_test = data_test[0].to(device), data_test[1].to(device)
                        outputs_test = model(images_test)
                        _, predicted = torch.max(outputs_test.data, 1)
                        total += labels_test.size(0)
                        correct += (predicted == labels_test).sum().item()
                
                print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
                test_acc.append(100 * correct / total)
                test_end = time.time()
                test_time += test_end - test_start

            # if cur_step >= args.total_steps:
            #     print((end_time - start_time)/args.total_steps)
            #     #exit the program
            #     exit()