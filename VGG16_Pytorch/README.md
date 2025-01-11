# VGG16_Pytorch

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
The network is trained using stochastic gradient descent(SGD). On TinyImageNet I train using batch size 32 for 30k epochs. The initial learning is set to 0.01, and is divided by 10 at 10k iteration and 20k interation of the total number of training epochs. Due to GPU memory constraints, this model is trained with a mini-batch size 32.

## Model Performance

<img width="467" alt="스크린샷 2025-01-12 오전 2 31 36" src="https://github.com/user-attachments/assets/2859d393-cb88-4e45-9075-a0439af9e172" />

| Error Metric    |  VGG16  |
|-----------------|---------|
| **Top-1 Error** |   10.2  |
| **Top-5 Error** |   3.5   |

## Reference
 - [VGG16 Paper](https://arxiv.org/pdf/1409.1556)
