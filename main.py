from __future__ import print_function
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from models import *
from dataset import NMNIST
from utils import choose_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

from tensorboardX import SummaryWriter

quantized_layers = []
def quantize_tensor(tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)

    #new_tensor = torch.round(scale * tensor)
    new_tensor = scale * tensor
    new_tensor = (new_tensor.round() - new_tensor).detach() + new_tensor
    return new_tensor, scale

def init_quantize_net(net):
    for name,m in net.named_modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            quantized_layers.append(m)
            m.weight.weight_float = m.weight.data.clone()

def quantize_layers(bitwidth,rescale=True):
    for i, layer in enumerate(quantized_layers):
        with torch.no_grad():
            quantized_w, scale_w=quantize_tensor(layer.weight.weight_float,bitwidth,False)
            layer.weight[...]= quantized_w/scale_w if rescale else quantized_w

class QuantSGD(torch.optim.SGD):
    """
    refactor torch.optim.SGD.step()
    For supporting the STE(Straight Through Estimator)
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if hasattr(p, 'weight_float'):
                    weight_data = p.weight_float
                else:
                    weight_data = p.data

                if p.grad is None:
                    continue
                # STE approximate function
                d_p = p.grad.data

                if weight_decay != 0:
                    # TODO: Explore the weight_decay
                    d_p.add_(weight_decay, weight_data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                weight_data.add_(-group['lr'], d_p)

        return loss

def weightsdistribute(model):
    print("================show every layer's weights distribute================")
    for key, value in model.named_parameters():
        print("================="+key+"=================")
        unique, count = torch.unique(value.detach(), sorted=True, return_counts= True)
        print(unique.shape)

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # necessary for general dataset: broadcast input
        data, _ = torch.broadcast_tensors(data, torch.zeros((args.timestep,) + data.shape)) 
        data = data.permute(1, 2, 3, 4, 0)

        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.quantize:
            quantize_layers(args.bit)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)],  Loss: {:.6f},  Acc: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data / args.timestep), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       correct, total, 100. * correct / total))
            if args.loss_writer:
                writer.add_scalar('Train Loss /batchidx', loss, batch_idx + len(train_loader) * epoch)
        
best_acc = 0
def test(args, model, device, test_loader, epoch, writer):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((args.timestep,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    if args.loss_writer:
        writer.add_scalar('Test Loss /epoch', test_loss, epoch)
        writer.add_scalar('Test Acc /epoch', acc, epoch)
        for i, (name, param) in enumerate(model.named_parameters()):
            if '_s' in name:
                writer.add_histogram(name, param, epoch)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    if acc > best_acc:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            state = {
                'model': model.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        else:
            print("no")
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('ckpt'):
            os.mkdir('ckpt')
            
        if args.quantize:
            torch.save(state, './ckpt/' + args.model + '_ckpt_q.pth')
            print('Saved in ./ckpt/' + args.model + '_ckpt_q.pth\n')
        else:
            torch.save(state, './ckpt/' + args.model + '_ckpt.pth')
            print('Saved in ./ckpt/' + args.model + '_ckpt.pth\n')
        best_acc = acc

    
def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 35))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('model', type=str,
                        help='network model type: see detail in folder ''models'' ')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset for model to train')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='continue the train')
    parser.add_argument('--loss_writer', '-lw', action='store_true', default=False,
                        help='For plot Tensorboard and see loss')
    
    parser.add_argument('--timestep', '-t', type=int, default=2,
                        help='parameter timestep of LIF neuron')
    parser.add_argument('--Vth', '-v', type=int, default=0.4,
                        help='parameter Vth of LIF neuron')
    parser.add_argument('--tau', '-ta', type=int, default=0.25,
                        help='parameter leaky tau of LIF neuron')

    parser.add_argument('--quantize', '-q', action='store_true', default=False,
                        help='QAT for snn')
    parser.add_argument('--bit', '-b', type=int, default=8,
                        help='bit num to quantize')
    
    parser.add_argument("--parallel", '-p', default = None ,type=str,
                    help='choose DP or DDP')
    parser.add_argument("--local_rank", type=int,
                        help='When there is a host slave situation in DDP,\
                        the host is local_ rank = 0')

    args = parser.parse_args()

    config_snn_param(args)
    #print(get_snn_param())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    
    if args.loss_writer:
        writer = SummaryWriter('./summaries/' + args.model + "/")
    else:
        writer = None
    
    if args.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1307], std=[0.3081])
                            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1307], std=[0.3081])
                            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    if args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = choose_model(args)
    model = model.to(device)

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('ckpt'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ckpt/' + args.model + '_ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        if not args.quantize:
            best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    if device == 'cuda' and torch.cuda.device_count() > 1 and args.parallel == 'DDP':
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
        model = DDP(model, find_unused_parameters=True)
    
    if args.quantize:
        print("========== quantize =============")
        init_quantize_net(model)
        quantize_layers(args.bit)
        
    # assert False
    optimizer = QuantSGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)
        #weightsdistribute(model)

    if args.loss_writer:
        writer.close()



if __name__ == '__main__':
    main()
