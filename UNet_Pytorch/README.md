# UNet_pytorch

* If you want to know about the information of FCN click here -> [Papaer Review - UNet](https://hoya9802.github.io/paper-review/UNet/)

## Environment of Implementation

### Version of Python
```shell
conda create -n "env name" python==3.8
```

### Version of Pytorch
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==10.1 -c pytorch
```
**Warning**: Recommend using more than version 2.0 of Pytorch because of one-hot encoding and permute compatibility issues!

### Installation of CUDA
```shell
conda install anaconda cudnn
```

## Dataset
### VOC dataset
1464 training set and train gt / 1449 test set and test gt

## Model
The [network.py](https://github.com/hoya9802/DL_Pytorch/blob/main/UNet_Pytorch/network.py) include UNet architecture

## Model Performance

<img width="660" alt="스크린샷 2025-01-12 오전 2 05 55" src="https://github.com/user-attachments/assets/0da94823-5ff2-4a4b-b2e4-47336fe7a80b" />

In Figure 1, we can see that U-Net provides more detailed segmentation compared to the previously conducted FCN-8s on the same examples. This is likely because U-Net uses a method where features obtained from each layer in the encoding stage are combined with the corresponding layers in the decoding stage. As a result, there is less loss of image values during the decoding process compared to FCN-8s. Apart from the examples shown above, it was observed that U-Net generally performs well in segmenting single objects. However, the result from the third row in the lower part did not show any significant improvement over the previous method.

## SmileNet Suggestion

<img width="661" alt="스크린샷 2025-01-12 오전 2 06 13" src="https://github.com/user-attachments/assets/8c2b7043-0da8-45e3-9f3c-41157415b399" />

In Figure 2, we observe that the image with Canny edges applied has values close to 0 in areas excluding the edges, with higher values only at the edges themselves. Since this can dilute the Canny edge values when passed into U-Net, we propose passing the Canny edge output through a shallow layer before connecting it to the skip connection of the U-Net using summation. This approach allows each layer to learn the Canny edges during the training process, which is expected to improve the accuracy of the output. Additionally, during the Canny edge extraction, a Gaussian filter is used to remove noise from the image, and since the Canny edge layer is shallow, normalization is not required. Finally, by adjusting the sigma value of the Canny edge and setting the ground truth of the training data accordingly, we anticipate that the model will be able to produce varied results depending on the desired sigma value of the Canny edge.
