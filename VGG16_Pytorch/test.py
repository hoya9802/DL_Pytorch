from network import *
from function import *
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
Device = "cuda" if torch.cuda.is_available() else 'cpu'
print(Device)

path = "/home/hoya9802/Downloads/tinyImageNet/"
test_path = path + 'test/'

model = VGG16(200)
model.load_state_dict(torch.load('/home/hoya9802/PycharmProjects/pythonProject/torchenv/VGG16/model/model_290000.pt'))
model.to(Device)
test_images, test_cls = load_image(test_path, path + 'test_gt.txt')

criterion = nn.CrossEntropyLoss()



model.eval()

test_loss = 0.0
correct = 0
total = 0

count = 0
for itest in range(test_images.shape[0]):
    if itest % 100 == 0:
        print('%d / %d' %(itest, test_images.shape[0]))

    test_image = (test_images[itest:itest+1, :, :] / 255.0) * 2 - 1 # [1,
    test_image = np.transpose(test_image, (0, 3, 1, 2))

    with torch.no_grad():
        pred = model(torch.from_numpy(test_image.astype(np.float32)).to(Device))

    pred = model(torch.from_numpy(test_images.astype(np.float32)).to(Device)) # [batch, 10] = [64, 10]
    cls_tensor = torch.tensor(test_cls, dtype=torch.long).to(Device)

    target = torch.tensor(test_cls[itest]).to(Device)
    loss = criterion(pred, target)

    test_loss += loss.item()

    pred = pred.cpu().numpy()
    pred = np.reshape(pred, 200)

    pred = np.argmax(pred)
    gt = test_cls[itest]

    if int(gt) == int(pred):
        count += 1

print('Accuracy : %.4f' %(count / test_images.shape[0] * 100))
average_loss = test_loss / test_images.shape[0]
print(f'Test Loss: {average_loss}')
