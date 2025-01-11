# Attention_pytorch

## Environment of Implementation

### Version of Python
 - conda create -n "env name" python==3.7

### Version of Pytorch
 - conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch

### Installation of CUDA
 - conda install anaconda cudnn

## Dataset
### TinyImageNet
This dataset consists 0.2m images for training, and 51k for test, from 200 classes. I didn't adopt any data augmentation, and the image sizes of the both training and test data are (128, 128, 3).

## Training
The network is trained using stochastic gradient descent(SGD). On TinyImageNet I train using batch size 32 for 30k epochs. The initial learning is set to 0.01 and a momentum of 0.9, and is divided by 10 at 10k iteration and 20k interation of the total number of training epochs. Due to GPU memory constraints, this model is trained with a mini-batch size 32.

## Models
The [network.py](https://github.com/hoya9802/DL_Pytorch/blob/main/Attention_Pytorch/network.py) include 3 Attention models (SE-ResNet-18, CBAM-ResNet-18, BAM-ResNet-18)

## Model Performance
epoch: 3000 / lr: 1e-2 / optimizer: SGD
| Error Metric    | ResNet-18 | SE-ResNet-18 | BAM-18 | CBAM-18 |
|-----------------|---------|---------|---------|---------|
| **Top-1 Error** | 50.05%   | 47.52%   |  42.46%   |  41.54%   |
| **Top-5 Error** | 26.55%   | 23.94%   |  19.98%   |  18.79%   |


## References
- [CBAM: Convolutional Block Attention Module (2018)](https://arxiv.org/pdf/1807.06521)
- [BAM: Bottleneck Attention Module (2018)](https://arxiv.org/pdf/1807.06514)
- [Squeeze-and-Excitation Networks (2017)](https://arxiv.org/pdf/1709.01507)
