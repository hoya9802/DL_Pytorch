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

model_save_path = '/Users/euntaeklee/torch_env/torch_class/Attention/model/'
path = "/Users/euntaeklee/torch_env/torch_class/data/tinyImageNet/"

print('load_zip')
z_train = zipfile.ZipFile(path + 'train.zip', 'r')
z_train_list = z_train.namelist()
train_cls = read_gt(path + 'train_gt.txt', len(z_train_list))
z_train_list, train_cls = zip_sort(z_train_list, train_cls)

z_test = zipfile.ZipFile(path + 'test.zip', 'r')
z_test_list = z_test.namelist()
test_cls = read_gt(path + 'test_gt.txt', len(z_test_list))
z_test_list, test_cls = zip_sort(z_test_list, test_cls)
print(len(z_test_list))
print(len(test_cls))

train_loss_history = []
# 2. build network  img ==> [128, 128, 3]

# model = ResNet18(num_class).to(Device)
# model = ResNet18_sq(num_class).to(Device)
# model = ResNet18_BAM(num_class).to(Device)
model = ResNet18_CBAM(num_class).to(Device)


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

    batch_img, batch_cls = Mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size=batch_size)
    # batch_img = [8, 128, 128, 3] -> [8, 3, 128, 128]
    batch_img = np.transpose(batch_img, (0, 3, 1, 2))

    # ---- training step
    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(Device)) # [batch, 10] = [64, 10]
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(Device)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    if it % 100 == 0:
        consum_time = time.time() - start_time
        train_loss_history.append(train_loss.item())
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

        t1_count = 0
        t5_count = 0

        num_test_img = len(z_test_list)

        for itest in range(num_test_img):
            if itest % 100 == 0:
                print('%d / %d' %(itest, num_test_img))

            img_temp = z_test.read(z_test_list[itest])
            img_temp = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
            img_temp = img_temp.astype(np.float32)

            test_image = (img_temp / 255.0) * 2 - 1
            test_image = np.transpose(test_image, (2, 0, 1))[np.newaxis]

            with torch.no_grad():
                pred = model(torch.from_numpy(test_image.astype(np.float32)).to(Device))

            pred = pred.numpy() # [1, 200]
            pred = np.reshape(pred, num_class) # [200]

            gt = test_cls[itest]

            # --- top 1 & top 5
            for ik in range(5):
                max_index = np.argmax(pred)
                if int(gt) == int(max_index):
                    t5_count += 1

                    if ik == 0:
                        t1_count += 1

                pred[max_index] = -9999



            # top5: [0.2, -0.1, 0.6, 0.7, ....] -> top 5 고르는 것 그중 하나라도 gt랑 정답이 같으면 맞췄다고 생각

        print('top-1 : %.4f top-5 : %.4f \n' %(t1_count / num_test_img * 100, t5_count / num_test_img * 100))
        f = open('%s.txt' %(model_save_path + 'acuuracy'), 'a+')
        f.write('top-1 : %.4f top-5 : %.4f \n' %(t1_count / num_test_img * 100, t5_count / num_test_img * 100))
        f.close()

plt.figure(figsize=(12, 5))
plt.plot(train_loss_history)
plt.title('Training Set Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('/Users/euntaeklee/torch_env/torch_class/Attention/train_loss.png')
plt.show()