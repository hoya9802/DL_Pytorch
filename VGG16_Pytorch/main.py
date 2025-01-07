import numpy as np
import torch
from function import *
from network import *
import os
import time
import matplotlib.pyplot as plt

Device = torch.device("mps")
print(Device)

num_class = 200
batch_size = 64

model_save_path = '/Users/euntaeklee/torch_env/torch_class/VGG16/model/'
path = "/Users/euntaeklee/torch_env/torch_class/data/tinyImageNet/"

train_path = path + 'train/'
test_path = path + 'test/'
train_images, train_cls = load_image(train_path, path + 'train_gt.txt')
test_images, test_cls = load_image(test_path, path + 'test_gt.txt')

train_loss_history = []

# 2. build network  img ==> [128, 128, 3]
model = VGG16(num_class).to(Device)

loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
num_iter = 300000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1.0)

start_time = time.time()
for it in range(num_iter):
    if it >= 100000 and it < 200000:
        optimizer.param_groups[0]['lr'] = 0.001
    if it >= 200000:
        optimizer.param_groups[0]['lr'] = 0.0001

    batch_img, batch_cls = Mini_batch_training(train_images, train_cls, batch_size=batch_size)
    # batch_img = [8, 128, 128, 3] -> [8, 3, 128, 128]
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    # ---- training step
    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(Device)) # [batch, 10] = [64, 10]
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(Device)

    train_loss = loss(pred, cls_tensor)
    train_loss_history.append(train_loss.item())
    train_loss.backward()
    optimizer.step()

    if it % 100 == 0:
        consum_time = time.time() - start_time
        print('iter: %d   train loss: %.5f   lr: %.5f   time: %.4f'
              %(it, train_loss.item(), optimizer.param_groups[0]['lr'], consum_time))
        model.eval()
        start_time = time.time()

    if it % 10000 == 0 and it != 0:
        print('SAVING MODEL')
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path)

        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        print('SAVING MODEL FINISH')

        count = 0
        for itest in range(test_images.shape[0]):
            if itest % 100 == 0:
                print('%d / %d' %(itest, test_images.shape[0]))

            test_image = (test_images[itest:itest+1, :, :] / 255.0) * 2 - 1 # [1,
            test_image = np.transpose(test_image, (0, 3, 1, 2))

            with torch.no_grad():
                pred = model(torch.from_numpy(test_image.astype(np.float32)).to(Device))

            pred = pred.numpy() # [1, 200]
            pred = np.reshape(pred, num_class) # [200]

            pred = np.argmax(pred)
            gt = test_cls[itest]

            if int(gt) == int(pred):
                count += 1

        print('Accuracy : %.4f' %(count / test_images.shape[0] * 100))

plt.figure(figsize=(12, 5))
plt.plot(train_loss_history)
plt.title('Training Set Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

plt.savefig('./train_loss.png')