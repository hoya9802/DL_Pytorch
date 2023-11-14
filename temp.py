import numpy as np
import cv2


VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "potted plant", "sheep", "sofa", "train", "tv/monitor", "edge",
               ]

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0], [192, 224, 224],
                ]

# ----- 1. color map to index
gt = cv2.imread('./images/gt.png')
gt_index = np.zeros(shape=(gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
gt_re = np.zeros(shape=(gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

# code = VOC_COLORMAP[0]
# aa = np.where(np.all(gt == code, axis=-1))
# temp = np.where(np.all(gt == code, axis=-1), 0, 255)
# temp = temp.astype(np.uint8)

for it in range(len(VOC_COLORMAP)):
    code = VOC_COLORMAP[it]

    if it != len(VOC_COLORMAP) - 1:
        gt_index[np.where(np.all(gt == code, axis=-1))] = it
    else:
        gt_index[np.where(np.all(gt == code, axis=-1))] = 255



cv2.imshow('gt', gt)
cv2.imshow('temp', gt_index)
# cv2.waitKey(-1)

# ---------- index to color
for it in range(len(VOC_COLORMAP)):
    code = VOC_COLORMAP[it]

    if it != len(VOC_COLORMAP) - 1:
        gt_re[np.where(np.all(gt_index == it, axis=-1))] = code
    else:
        gt_re[np.where(np.all(gt_index == it, axis=-1))] = code


cv2.imshow('gt_re', gt_re)

cv2.waitKey(-1)
