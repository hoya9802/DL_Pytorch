# ResNet_Pytorch

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

## Model
The [network.py](https://github.com/hoya9802/DL_Pytorch/blob/main/ResNet_Pytorch/network.py) include 5 types of ResNet models (ResNet-18, 34, 50, 101, 152)

## Model Performance
| Error Metric    | ResNet-18 | ResNet-34 | ResNet-50 |
|-----------------|---------|---------|---------|
| **Top-1 Error** |  50.05  |  48.78  |  47.99  |
| **Top-5 Error** |  26.55  |  25.34  |  23.63  |

## References
 - [ResNet paper](https://arxiv.org/pdf/1512.03385)
