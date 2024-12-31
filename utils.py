import os
import csv
import numpy as np
import scipy.io as sio
import torch
import cv2
import shutil

root = 'multicue/'
dst_root = 'data/'
path = root + 'images'
gt_path = root + 'ground-truth/images/edges'
dst_path = dst_root + 'images/'
dst_gt_path = dst_root + 'ground-truth/'


def split_train_test():
    gt_list = []
    for gt in os.listdir(root + 'ground-truth/hdf5'):
        gt_list.append(gt[:-3])
    np.random.shuffle(gt_list)
    offset = int(len(gt_list) * 0.6)
    train_data = gt_list[:offset]
    test_data = gt_list[offset:]
    return train_data, test_data


def data_pre_processing(img_data):
    for name in img_data:
        img_file = path + '/' + name + '.png'
        img = cv2.imread(img_file)
        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst_path + name + '.png', img)
        res_gt = np.zeros(img.shape)
        for i in range(1, 7):
            gt_file = gt_path + '/' + name + '.{}.png'.format(i)
            gt = cv2.imread(gt_file)
            gt = cv2.resize(gt, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            res_gt += np.array(gt)
        cv2.imwrite(dst_gt_path + name + '.png', res_gt)
        print('{} is ready'.format(name))


def crop_image(img, angle):
    h, w, _ = img.shape
    if angle == 0.0 or angle == 180.0:
        return img
    elif angle == 90.0 or angle == 270.0:
        return img[:, w // 2 - h // 2:w // 2 + h // 2]
    else:
        return img[h // 2 - h // 4:h // 2 + h // 4, w // 2 - w // 4:w // 2 + w // 4]


def data_augmentation(train_data, test_data):
    aug_root = dst_root + 'train/'
    for scale_size in [1, 0.5, 0.75, 1.25, 1.5]:
        aug_path1 = aug_root + 'aug_data' + ('' if scale_size == 1 else '_scale_{}'.format(scale_size))
        aug_path_gt1 = aug_root + 'aug_gt' + ('' if scale_size == 1 else '_scale_{}'.format(scale_size))
        if not os.path.exists(aug_path1):
            os.mkdir(aug_path1)
        if not os.path.exists(aug_path_gt1):
            os.mkdir(aug_path_gt1)
        for angle in np.arange(0.0, 356, 22.5):
            aug_path2_0 = aug_path1 + '/' + '{}'.format(angle) + '_0'
            aug_path_gt2_0 = aug_path_gt1 + '/' + '{}'.format(angle) + '_0'
            aug_path2_1 = aug_path1 + '/' + '{}'.format(angle) + '_1'
            aug_path_gt2_1 = aug_path_gt1 + '/' + '{}'.format(angle) + '_1'
            if os.path.exists(aug_path2_0):
                shutil.rmtree(aug_path2_0)
            os.mkdir(aug_path2_0)
            if os.path.exists(aug_path_gt2_0):
                shutil.rmtree(aug_path_gt2_0)
            os.mkdir(aug_path_gt2_0)
            if os.path.exists(aug_path2_1):
                shutil.rmtree(aug_path2_1)
            os.mkdir(aug_path2_1)
            if os.path.exists(aug_path_gt2_1):
                shutil.rmtree(aug_path_gt2_1)
            os.mkdir(aug_path_gt2_1)
            for data in train_data:
                img_file = dst_path + data + '.png'
                gt_file = dst_gt_path + data + '.png'
                img = cv2.imread(img_file)
                gt = cv2.imread(gt_file)
                h, w, _ = img.shape
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
                gt = cv2.warpAffine(gt, M, (w, h))
                img = crop_image(img, angle)  # 裁剪（去黑边）
                gt = crop_image(gt, angle)
                img = cv2.resize(img, dsize=None, fx=scale_size, fy=scale_size, interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, dsize=None, fx=scale_size, fy=scale_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(aug_path2_0 + '/' + data + '.png', img)
                cv2.imwrite(aug_path_gt2_0 + '/' + data + '.png', gt)
                cv2.imwrite(aug_path2_1 + '/' + data + '.png', cv2.flip(img, 1))
                cv2.imwrite(aug_path_gt2_1 + '/' + data + '.png', cv2.flip(gt, 1))
                print(data + ' : {}, {}'.format(scale_size, angle))

    test_root = dst_root + 'test/'
    if os.path.exists(test_root[:-1]):
        shutil.rmtree(test_root[:-1])
    os.mkdir(test_root[:-1])
    for data in test_data:
        img_file = dst_path + data + '.png'
        img = cv2.imread(img_file)
        cv2.imwrite(test_root + data + '.png', img)


def create_data_file(train_data, test_data):
    train_f = open('data//train_pair.lst', 'w', newline='')
    test_f = open('data/test.lst', 'w', newline='')
    train_write = csv.writer(train_f)
    test_write = csv.writer(test_f)
    train_root = dst_root + 'train/'
    for scale_size in [1, 0.5, 0.75, 1.25, 1.5]:
        aug_path1 = train_root + 'aug_data' + ('' if scale_size == 1 else '_scale_{}'.format(scale_size))
        aug_path_gt1 = train_root + 'aug_gt' + ('' if scale_size == 1 else '_scale_{}'.format(scale_size))
        for angle in np.arange(0.0, 356, 22.5):
            aug_path2_0 = aug_path1 + '/' + '{}'.format(angle) + '_0'
            aug_path_gt2_0 = aug_path_gt1 + '/' + '{}'.format(angle) + '_0'
            aug_path2_1 = aug_path1 + '/' + '{}'.format(angle) + '_1'
            aug_path_gt2_1 = aug_path_gt1 + '/' + '{}'.format(angle) + '_1'
            for data in train_data:
                train_write.writerow([aug_path2_0 + '/' + data + '.png' + ' ' + aug_path_gt2_0 + '/' + data + '.png'])
                train_write.writerow([aug_path2_1 + '/' + data + '.png' + ' ' + aug_path_gt2_1 + '/' + data + '.png'])
    test_path = dst_root + 'test'
    for dir in os.listdir(test_path):
        test_write.writerow([test_path + '/' + dir])
    train_f.close()
    test_f.close()


def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params = model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)


class Averagvalue(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    train_data, test_data = split_train_test()
    data_pre_processing(train_data + test_data)
    data_augmentation(train_data, test_data)
    create_data_file(train_data, test_data)
