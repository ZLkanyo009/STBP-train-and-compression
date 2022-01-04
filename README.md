# Train SNN with STBP in fp32 and low bit(quantize)

Spiking neural network (SNN), compared with depth neural network (DNN), has faster processing speed, lower energy consumption and more biological interpretability, which is expected to approach Strong AI.

STBP is a way to train SNN with datasets by Backward propagation.Using this Repositories allows you to train SNNS with STBP and quantize SNNS with QAT to deploy to neuromorphological chips like Loihi and Tianjic.


## Usage

### Download via GitHub:

```
git clone https://github.com/ZLkanyo009/STBP-train-and-compression.git
```

### example to define SNN_layers like ANN_layers

Convert layer to spatiotemporal layer:
```
conv = nn.Conv2d(...)
conv_s = tdLayer(conv)
```

Define LIF activation function just like Relu:
```
spike = LIFSpike()
```

In the forward function, replace the activation function of each layer with LIF activation function, and replace the calls such as `conv()` with `conv_ s()`, then SNN_layers definition is completed.Finally, we use Frequency Coding to decode SNN's output like `out = torch.sum(x, dim=2) / steps`
```
def forward(self, x):
    x = self.conv1_s(x)
    x = self.spike(x)
    x = self.pool1_s(x)
    x = self.spike(x)
    x = x.view(x.shape[0], -1, x.shape[4])
    x = self.fc1_s(x)
    x = self.spike(x)
    out = torch.sum(x, dim=2) / steps
    return out
```

If BN layer is required:

```
bn = nn.BatchNorm2d(...)
bn = tdBatchNorm(...)
conv_s = tdLayer(conv, bn)
```

### Training Fp32 Model
```
# Start training fp32 model with: 
# model_name can be ResNet18, CifarNet, ...
python main.py ResNet18 --dataset CIFAR10

# training with DDP:
python -m torch.distributed.launch main.py ResNet18 --local_rank 0 --dataset CIFAR10 --p DDP

# You can manually config the training with: 
python main.py ResNet18 --resume --lr 0.01
```

### Training Quantize Model
```
# Start training quantize model with: 
# model_name can be ResNet18, CifarNet, ...
python main.py ResNet18 --dataset CIFAR10 -q

# training with DDP:
python -m torch.distributed.launch main.py ResNet18 --local_rank 0 --dataset CIFAR10 -q --p DDP

# You can manually config the training with: 
python main.py ResNet18 -q --resume --bit 4 --lr 0.01

```

## Accuracy
All SNN run in timesteps = 2.
| Model            | Acc.(fp32) | Acc.(8 bit quantize) | Acc.(4 bit quantize) |
| ---------------- | ---------- | -------------------- | -------------------- |
| MNISTNet         | 97.96%     | 97.57%               | 97.56%               |
| ResNet18         | 84.40%     | 84.23%               | 83.61%               |
| ResNet18         | 84.40%     | 84.23%               | 83.61%               |

## About STBP

- [Zheng, H., Wu, Y., Deng, L., Hu, Y., & Li, G. (2020). Going Deeper With Directly-Trained Larger Spiking Neural Networks. *arXiv preprint arXiv:2011.05280*.](https://arxiv.org/pdf/2011.05280)
- [Wu, Y., Deng, L., Li, G., Zhu, J., Xie, Y., & Shi, L. (2019, July). Direct training for spiking neural networks: Faster, larger, better. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, pp. 1311-1318).](https://www.aaai.org/ojs/index.php/AAAI/article/view/3929/3807)
- [Wu, Y., Deng, L., Li, G., Zhu, J., & Shi, L. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. *Frontiers in neuroscience*, *12*, 331.](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)