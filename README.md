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
The network is trained using stochastic gradient descent(SGD). On TinyImageNet I train using batch size 32 for 30k epochs. The initial learning is set to 0.01, and is divided by 10 at 10k iteration and 20k interation of the total number of training epochs. Due to GPU memory constraints, this model is trained with a mini-batch size 32.
