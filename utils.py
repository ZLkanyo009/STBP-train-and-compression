from models import *

def choose_model(args):
    if args.model == "ResNet18":
        return ResNet18()
    if args.model == "NMNISTNet":
        return NMNISTNet()
    if args.model == "MNISTNet":
        return MNISTNet()
    if args.model == "CifarNet":    
        return CifarNet()
    
    else:
        print("error model name, not support: ", args.model)
        exit(0)