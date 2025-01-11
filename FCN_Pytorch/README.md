# FCN_pytorch

* If you want to know about the information of FCN click here -> [Papaer Review - FCN](https://hoya9802.github.io/paper-review/FCN/)

## Environment of Implementation

### Version of Python
 - conda create -n "env name" python==3.8

### Version of Pytorch
 - conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch
<span style="color:red">**(Recommend using more than version 2.0 of Pytorch because of one-hot encoding and permute )compatibility issues!**</span>

### Installation of CUDA
 - conda install anaconda cudnn

## Dataset
### VOC dataset
1464 training set and train gt / 1449 test set and test gt

## Model
The [network.py](https://github.com/hoya9802/DL_Pytorch/blob/main/FCN_Pytorch/network.py) include FCN_8s Model

## Model Performance

<img width="632" alt="스크린샷 2025-01-12 오전 1 45 24" src="https://github.com/user-attachments/assets/af2c0cf1-a6b0-425f-8d83-5912e4ab709a" />

In Figure 1, we observed that while some objects showed relatively good segmentation results, in most cases the performance was suboptimal. Specifically, for closely interacting objects (third row), the performance was even worse, demonstrating a significant drop in accuracy.

## Proposals for Improvement

<img width="577" alt="스크린샷 2025-01-12 오전 1 46 11" src="https://github.com/user-attachments/assets/d8a35d0c-16d4-42fb-b801-b483331eb317" />

In the previous decoding stage using up-sampling, it was considered advantageous to retain as much of the image information as possible. Therefore, after applying up-sampling using transposed convolution, we applied feature norm (instead of batch norm) to the up-sampled feature map, followed by a Standard Gaussian Distribution. Then, before down-sampling, we applied vector-wise multiplication to the input. Afterward, the feature map from the first layer is cropped to match the size of the up-sampled feature map, and concatenation is performed. This process corrects the non-linear values from the encoding process to linearity during up-sampling. Additionally, since some values may be lost during compression in the learning process, cropping the first layer and concatenating it helps compensate for the values lost during compression. As a result, the fine features from the original image are linked to the down-sampled, coarse image, enabling more accurate dense predictions and potentially yielding better results (see Figure 2). However, if the number of filters in the first layer is too high, the values might become overly mixed. Without mitigating the increased computational load using bottleneck structures, the results could deteriorate further.
