import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from train import *
from train2 import *
from sklearn.metrics import precision_recall_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = RCF()
net.eval()
# net.load_state_dict(torch.load('models2/model_epoch500_10180801.pth'), strict=True)
net.load_state_dict(torch.load('checkpoint_epoch18.pth')['state_dict'], strict=True)
net = net.to(device)

img_path = '2092.jpg'
# img_path = '1.jpg'
img = Image.open(img_path)
x = np.array(img, dtype=np.float32)
# x = prepare_image_PIL(x)
x = np.transpose(x, (2, 0, 1))
x = torch.from_numpy(x)
x = x.unsqueeze(0)
_, _, H, W = x.shape
x = x.to(device)
results = net(x)
result = torch.squeeze(results[-1].detach()).cpu().numpy()
res = Image.fromarray((result * 255).astype(np.uint8))
contrast = Image.new(img.mode, (W * 2, H))
contrast.paste(img, box=(0, 0))
contrast.paste(res, box=(W, 0))
contrast.show()

lb = np.array(Image.open('2092.png'))
if lb.ndim == 3:
    lb = np.squeeze(lb[:, :, 0])
# lb[lb != 0] = 1
lb[lb == 0] = 0
lb[np.logical_and(lb > 0, lb < 128)] = 2
lb[lb >= 128] = 1
true = []
predict = []
for i in range(lb.shape[0]):
    for j in range(lb.shape[1]):
        if lb[i][j] != 2:
            true.append(lb[i][j])
            predict.append(result[i][j])
# pre, rec, _ = precision_recall_curve(lb.reshape(-1), result.reshape(-1))
pre, rec, _ = precision_recall_curve(true, predict)
plt.plot(pre, rec)
plt.show()

