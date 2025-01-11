# PSPNet_pytorch

## Environment of Implementation
pretrained_network.py was pretrained by resnet50

### Version of Python
```shell
conda create -n "env name" python==3.8
```
### Version of Pytorch
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch
```

**Warning:** Recommend using more than version 2.0 of Pytorch because of one-hot encoding and permute compatibility issues!

### Installation of CUDA
```shell
conda install anaconda cudnn
```
## Dataset
### VOC dataset
1464 training set and train gt / 1449 test set and test gt

## Model
The [network.py](https://github.com/hoya9802/DL_Pytorch/blob/main/PSPNet_Pytorch/network.py) include PSPNet and PSPModule

## Model Performance

<img width="511" alt="스크린샷 2025-01-12 오전 1 20 28" src="https://github.com/user-attachments/assets/136dbee7-7539-45a2-a2f5-03b9d12bed27" />

Looking at Figure 1, previous models such as FCN and U-Net performed very well in segmenting single objects with distinct shapes, like airplanes. However, they struggled to predict humans or objects interacting with humans. In contrast, PSP-Net successfully distinguishes not only humans but also objects interacting with them, demonstrating its ability to accurately segment such scenarios.

## Comparison Result

<img width="514" alt="스크린샷 2025-01-12 오전 1 20 58" src="https://github.com/user-attachments/assets/d9992796-3a6d-496b-8157-e727d8824c9d" />

Compared to previous models, we observed that PSP-Net extracts more objects with greater detail across all images. Although the images in the first row suggest that PSP-Net is less fine-tuned, it is important to note that FCN-8s was trained for 150,000 epochs, U-Net for 100,000 epochs, while PSP-Net was only trained for 20,000 epochs. Therefore, since FCN-8s was trained 7.5 times longer than PSP-Net, we believe that if PSP-Net undergoes sufficient training, it could achieve even more detailed results, similar to FCN-8s (See Figure 2.).

## Reference
 - [PSPNet](https://arxiv.org/pdf/1612.01105)
