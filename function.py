import cv2
import os
import numpy as np
import zipfile
import re

def Mini_batch_training(train_img, train_cls, batch_size):
    batch_img = np.zeros((batch_size, 128, 128, 3))
    batch_cls = np.zeros(batch_size)

    # train_img = [200xxx, 128, 128, 3]
    rand_num = np.random.randint(0, train_img.shape[0], size=batch_size) # [2, 19, 77, 10]

    # pixel normalization : 0 - 255 / -1 - 1
    for it in range(batch_size):
        temp = rand_num[it]
        batch_img[it, :, :] = (train_img[temp, :, :, :] / 255.0) * 2 - 1 # (0 ~ 1) x 2 -> 0 ~ 2 -> -1 - 1
        batch_cls[it] = train_cls[temp]

    return batch_img, batch_cls

def read_gt(gt_txt, num_img):
    cls = np.zeros(num_img)
    f = open(gt_txt, 'r')
    lines = f.readlines()
    for it in range(len(lines)):
        cls[it] = int((lines[it])[:-1]) - 1 # 0 ~ 199


    f.close()

    return cls

def Mini_batch_training_zip(z_file, z_file_list, train_cls, batch_size):
    batch_img = np.zeros((batch_size, 128, 128, 3))
    batch_cls = np.zeros(batch_size)

    # train_img = [20xxxx, 128, 128, 3]
    rand_num = np.random.randint(0, len(z_file_list), size=batch_size)

    # pixel normalization : 0~255 -> 0 ~ 1 / -1 ~ 1
    for it in range(batch_size):
        temp = rand_num[it]
        img_temp = z_file.read(z_file_list[temp])
        img_temp = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
        img_temp = img_temp.astype(np.float32)

        batch_img[it, :, :] = (img_temp / 255.0) * 2 - 1 # (0 ~ 1) x 2 -> 0 ~ 2 -> -1 ~ 1
        batch_cls[it] = train_cls[temp]

    return batch_img, batch_cls

def zip_sort(data, data_cls):
    data = sorted(data[1:], key=lambda x: int(re.split('[/.]', x)[1]))
    data_cls = data_cls[:-1]

    return data, data_cls