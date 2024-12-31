#!/user/bin/python
# coding=utf-8
import time
import pandas as pd
import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import optim
import cv2
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
from dataset import *
from functions import *
from net2 import My_RCF
from utils import load_vgg16pretrain, Averagvalue

# lr = 5.00000000000001e-10
lr = 1e-7
momentum = 0.9
# weight_decay = 1e-8
weight_decay = 0.0002
begin_epoch = 0
end_epoch = 10
iter_size = 10
print_freq = 300


def main():
    # dataset
    train_dataset = Dataset(split="train")
    test_dataset = Dataset(split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2, drop_last=True, shuffle=False)
    with open('data/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [i.split(' ')[0].split('/')[-1] for i in test_list]

    # model
    net = My_RCF()
    net = net.cuda()
    # net.apply(weights_init)
    # load_vgg19pretrain(net)
    # net.load_state_dict(torch.load('models2/model_epoch500_10180801.pth'), strict=True)

    # tune lr
    global lr
    net_parameters_id = {}
    for pname, p in net.named_parameters():
        if pname in ['conv1_1.weight', 'conv1_2.weight',
                     'conv2_1.weight', 'conv2_2.weight',
                     'conv3_1.weight', 'conv3_2.weight', 'conv3_3.weight', 'conv3_4.weight',
                     'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight', 'conv4_4.weight']:
            # print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias', 'conv1_2.bias',
                       'conv2_1.bias', 'conv2_2.bias',
                       'conv3_1.bias', 'conv3_2.bias', 'conv3_3.bias', 'conv3_4.bias',
                       'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias', 'conv4_4.bias']:
            # print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight', 'conv5_4.weight']:
            # print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias', 'conv5_4.bias']:
            # print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight', 'conv1_2_down.weight',
                       'conv2_1_down.weight', 'conv2_2_down.weight',
                       'conv3_1_down.weight', 'conv3_2_down.weight', 'conv3_3_down.weight', 'conv3_4_down.weight',
                       'conv4_1_down.weight', 'conv4_2_down.weight', 'conv4_3_down.weight', 'conv4_4_down.weight',
                       'conv5_1_down.weight', 'conv5_2_down.weight', 'conv5_3_down.weight', 'conv5_4_down.weight']:
            # print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias', 'conv1_2_down.bias',
                       'conv2_1_down.bias', 'conv2_2_down.bias',
                       'conv3_1_down.bias', 'conv3_2_down.bias', 'conv3_3_down.bias', 'conv3_4_down.bias',
                       'conv4_1_down.bias', 'conv4_2_down.bias', 'conv4_3_down.bias', 'conv4_4_down.bias',
                       'conv5_1_down.bias', 'conv5_2_down.bias', 'conv5_3_down.bias', 'conv5_4_down.bias']:
            # print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                       'score_dsn4.weight', 'score_dsn5.weight']:
            # print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias',
                       'score_dsn4.bias', 'score_dsn5.bias']:
            # print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            # print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

    optimizer = torch.optim.SGD([
        {'params': net_parameters_id['conv1-4.weight'], 'lr': lr * 1, 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv1-4.bias'], 'lr': lr * 2, 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight'], 'lr': lr * 10, 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv5.bias'], 'lr': lr * 20, 'weight_decay': 0.},
        {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': lr * 0.1, 'weight_decay': weight_decay},
        {'params': net_parameters_id['conv_down_1-5.bias'], 'lr': lr * 0.2, 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr * 0.01, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': lr * 0.02, 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight'], 'lr': lr * 0.001, 'weight_decay': weight_decay},
        {'params': net_parameters_id['score_final.bias'], 'lr': lr * 0.002, 'weight_decay': 0.},
    ], lr=lr, momentum=momentum, weight_decay=weight_decay)

    # 训练
    train_loss = []
    train_loss_detail = []
    for epoch in range(begin_epoch, end_epoch):
        tr_avg_loss, tr_detail_loss = train(train_dataloader, net, optimizer, epoch)
        test(test_dataloader, net, test_list, 'test_result')
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss
        print('epoch: {}, loss: {}'.format(epoch + 1, tr_avg_loss))
        torch.save(net.state_dict(), 'models/model_epoch{}_{}.pth'.format(epoch + 1, time.strftime("%m%d%H%M")))
        # if (epoch + 1) % 100 == 0:
        #     lr = lr / 10
        #     optimizer = torch.optim.SGD([
        #         {'params': net_parameters_id['conv1-4.weight'], 'lr': lr * 1, 'weight_decay': weight_decay},
        #         {'params': net_parameters_id['conv1-4.bbegin_epoch + 1, end_epoch + 1ias'], 'lr': lr * 2, 'weight_decay': 0.},
        #         {'params': net_parameters_id['conv5.weight'], 'lr': lr * 10, 'weight_decay': weight_decay},
        #         {'params': net_parameters_id['conv5.bias'], 'lr': lr * 20, 'weight_decay': 0.},
        #         {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': lr * 0.1, 'weight_decay': weight_decay},
        #         {'params': net_parameters_id['conv_down_1-5.bias'], 'lr': lr * 0.2, 'weight_decay': 0.},
        #         {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr * 0.01, 'weight_decay': weight_decay},
        #         {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': lr * 0.02, 'weight_decay': 0.},
        #         {'params': net_parameters_id['score_final.weight'], 'lr': lr * 0.001, 'weight_decay': weight_decay},
        #         {'params': net_parameters_id['score_final.bias'], 'lr': lr * 0.002, 'weight_decay': 0.},
        #     ], lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss_dt = pd.DataFrame(train_loss, index=range(begin_epoch + 1, end_epoch + 1), columns=['epoch', 'loss'])
    loss_dt.to_csv('{}loss {}-{}.xlsx'.format(time.strftime("%m%d%H%M"), begin_epoch + 1, end_epoch))

    # plt.plot(train_loss)
    # plt.savefig("train_loss{}.png".format(time.strftime("%m%d%H%M")), dpi=240)


def train(train_loader, model, optimizer, epoch):
    losses = Averagvalue()
    # switch to train mode
    model.train()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + cross_entropy_loss_RCF(o, label)
        loss = loss / iter_size
        loss.backward()
        counter += 1
        if counter == iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())

        if i % print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch + 1, end_epoch, i, len(train_loader)) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)
            print(info)

    return losses.avg, epoch_loss


def test(test_loader, model, test_list, save_dir):
    model.eval()
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]
        filename = test_list[idx].split('.')[0]
        torchvision.utils.save_image(1 - results_all, join(save_dir, "%s.jpg" % filename))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    main()
