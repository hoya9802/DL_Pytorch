import cv2
import os
import numpy as np

def load_image(path, gt_txt):
    # --- load image
    img_names = os.listdir(path)
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
    print(len(img_names))
    num_img = len(img_names)

    imgs = np.zeros((num_img, 128, 128, 3), dtype=np.uint8) # [B, H, W, C]

    img_count = 0
    for it in range(num_img):
        if it % 1000 == 0:
            print('%d / %d' %(it, num_img))

        img_temp = cv2.imread(path + img_names[it])
        imgs[img_count, :, :, :] = img_temp
        img_count += 1

    # --- load class
    cls = np.zeros(num_img)
    f = open(gt_txt, "r")
    lines = f.readlines()
    for it in range(len(lines)):
        cls[it] = int((lines[it])[:-1]) - 1 # 0 ~ 199

    f.close()

    return imgs, cls

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
