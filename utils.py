from models import *

def choose_model(args):
    if args.model == "ResNet18":
        if args.dataset == "CIFAR100":
            classes = 100
        elif args.dataset == "CIFAR10":
            classes = 10
        else:
            classes = 10
        return ResNet18(num_classes=classes)
    if args.model == "VGG19":
        return vgg19()
    if args.model == "NMNISTNet":
        return NMNISTNet()
    if args.model == "MNISTNet":
        return MNISTNet()
    if args.model == "CifarNet":    
        return CifarNet()
    
    else:
        print("error model name, not support: ", args.model)
        exit(0)