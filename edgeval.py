import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from train import *
from train2 import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def edgesEvalImg(E, G, K=99):
    thrs = np.linspace(1 / (1 + K), 1 - 1 / (1 + K), K)
    cntR, sumR, cntP, sumP = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    G = G.astype('uint8')
    for i in range(K):
        E1 = E.copy()
        E1[E1 >= thrs[i]] = 1
        E1[E1 < thrs[i]] = 0
        E1 = E1.astype('uint8')

        matchG = np.zeros(E.shape, dtype='float')
        allG = np.zeros(E.shape, dtype='float')
        for g in range(G.shape[0]):
            match = E1 & G[g]
            matchG += match
            allG += G[g]

        cntR[i] = matchG.sum()
        sumR[i] = allG.sum()
        cntP[i] = np.count_nonzero(matchG)
        sumP[i] = np.count_nonzero(E1)

    return thrs, cntR, sumR, cntP, sumP


def computeRPF(cntR, sumR, cntP, sumP):
    sumR[sumR == 0] = 1
    R = cntR / sumR
    sumP[sumP == 0] = 1
    P = cntP / sumP
    sumPR = P + R
    sumPR[sumPR == 0] = 1
    F = 2 * P * R / sumPR
    return R, P, F


def get_test_PRF(net, test_past='data/test'):
    testP, testR, testF = [], [], []
    for name in os.listdir(test_past):
        img = cv2.imread(test_past + '/' + name)
        x = np.array(img, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.to(device)
        results = net(x)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()

        gt = []
        for i in range(1, 7):
            g = cv2.imread('multicue/ground-truth/images/edges/' + name.split('.')[0] + '.{}.png'.format(i))
            g = cv2.resize(g, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            lb = np.array(g)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            lb[lb > 0] = 1
            gt.append(lb)

        [thrs, cntR, sumR, cntP, sumP] = edgesEvalImg(result, np.array(gt), 99)
        # print(cntR, sumR, cntP, sumP)
        R, P, F = computeRPF(cntR, sumR, cntP, sumP)
        testP.append(P)
        testR.append(R)
        testF.append(F)

    return np.array(testP), np.array(testR), np.array(testF)


def draw_PR(P, R, F, label):
    plt.plot(R.mean(axis=0), P.mean(axis=0), linewidth=3, label='[F={:.2f}]{}'.format(max(F.mean(axis=0)), label))


def draw_OIS(P, R, F):
    idx = F.argmax()
    x, y = idx // F.shape[1], idx % F.shape[1]
    plt.scatter(R[x][y], P[x][y], color='r', label='[F={:.2f}]OIS-F'.format(F[x][y]))


net = My_RCF()
net.eval()
net = net.to(device)

# net.load_state_dict(torch.load('model_epoch50_03180250.pth'), strict=True)
# testP, testR, testF = get_test_PRF(net)
# # draw_OIS(testP, testR, testF)
# draw_PR(testP, testR, testF, '50')
# net.load_state_dict(torch.load('model_epoch100_03181500.pth'), strict=True)
# testP, testR, testF = get_test_PRF(net)
# draw_PR(testP, testR, testF, '100')
# net.load_state_dict(torch.load('model_epoch150_03190118.pth'), strict=True)
# testP, testR, testF = get_test_PRF(net)
# draw_PR(testP, testR, testF, '150')
net.load_state_dict(torch.load('model_epoch200_03191403.pth'), strict=True)
testP, testR, testF = get_test_PRF(net)
draw_PR(testP, testR, testF, 'haze')
net.load_state_dict(torch.load('model_epoch200_10151339.pth'), strict=True)
testP, testR, testF = get_test_PRF(net)
draw_PR(testP, testR, testF, 'without haze')

# net2 = RCF()
# net2.eval()
# net2 = net2.to(device)
# net2.load_state_dict(torch.load('checkpoint_epoch18.pth')['state_dict'], strict=True)
# testP2, testR2, testF2 = get_test_PRF(net2)
# draw_PR(testP2, testR2, testF2, 'RCF')
# print(testR2.mean(axis=0)[49], testP2.mean(axis=0)[49])

plt.xticks(np.linspace(0, 1, 5))
plt.yticks(np.linspace(0, 1, 5))
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.show()

# net.load_state_dict(torch.load('model_epoch50_03180250.pth'), strict=True)
# testP1, testR1, testF1 = get_test_PRF(net)
# print(testR1.mean(axis=0)[49], testP1.mean(axis=0)[49])
# net.load_state_dict(torch.load('model_epoch100_03181500.pth'), strict=True)
# testP2, testR2, testF2 = get_test_PRF(net)
# print(testR2.mean(axis=0)[49], testP2.mean(axis=0)[49])
# net.load_state_dict(torch.load('model_epoch150_03190118.pth'), strict=True)
# testP3, testR3, testF3 = get_test_PRF(net)
# print(testR3.mean(axis=0)[49], testP3.mean(axis=0)[49])
# net.load_state_dict(torch.load('model_epoch200_03191403.pth'), strict=True)
# testP4, testR4, testF4 = get_test_PRF(net)
# print(testR4.mean(axis=0)[49], testP4.mean(axis=0)[49])
# net.load_state_dict(torch.load('model_epoch1000_10221158.pth'), strict=True)
# testP5, testR5, testF5 = get_test_PRF(net)
# print(testR5.mean(axis=0)[49], testP5.mean(axis=0)[49])

