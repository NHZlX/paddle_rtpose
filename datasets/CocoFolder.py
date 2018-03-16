import numpy as np
import os
import math
import json
import cv2
import pickle

import Mytransforms

def read_data_file(file_dir):
    with open(file_dir, 'r') as fp:
        lines = [line.strip()for line in fp.readlines()]
    return lines

def read_json_file(file_dir):
    """
        filename: JSON file
        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []

    for info in data:
        kpt = []
        center = []
        scale = []
        lists = info['info']
        for x in lists:
           kpt.append(x['keypoints'])
           center.append(x['pos'])
           scale.append(x['scale'])
        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
    fp.close()

    return kpts, centers, scales

def generate_heatmap(heatmap, kpt, stride, sigma):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    length = len(kpt[0])
    # the num of the person in one pic
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] == 0:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j + 1] += math.exp(-dis)
                    if heatmap[h][w][j + 1] > 1:
                        heatmap[h][w][j + 1] = 1
    return heatmap

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] == 0 or kpts[j][b][2] == 0:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1
    return vector

class CocoFolder:
    def __init__(self, data_path, file_dir, stride, transformer=None):
        self.base_path = data_path 
        self.img_list = read_data_file(file_dir[0])
        self.mask_list = read_data_file(file_dir[1])
        self.kpt_list, self.center_list, self.scale_list = read_json_file(file_dir[2])
        self.stride = stride
        self.transformer = transformer
        self.vec_pair = [[2,3,5,6,8,9, 11,12,0,1,1, 1,1,2, 5, 0, 0, 14,15],
                         [3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,14,15,16,17]] # different from openpose
        self.theta = 1.0
        self.sigma = 7.0

    def reader(self):
        length = len(self.img_list)

        for index in xrange(length):
            try:
                img_path = os.path.join(self.base_path, self.img_list[index])
                img = np.array(cv2.imread(img_path), dtype=np.float32)
                mask_path = self.mask_list[index]
                mask = pickle.load(open(mask_path))
            except:
                continue

            mask = np.array(mask, dtype=np.float32)

            kpt = self.kpt_list[index]
            center = self.center_list[index]
            scale = self.scale_list[index]

            img, mask, kpt, center = self.transformer(img, mask, kpt, center, scale)

            height, width, _ = img.shape

            mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape((height / self.stride, width / self.stride, 1))

            heatmap = np.zeros((height / self.stride, width / self.stride, len(kpt[0]) + 1), dtype=np.float32)
            heatmap = generate_heatmap(heatmap, kpt, self.stride, self.sigma)
            heatmap[:,:,0] = 1.0 - np.max(heatmap[:,:,1:], axis=2) # for background
            heatmap = heatmap * mask

            vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
            cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0])), dtype=np.int32)

            vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
            vecmap = vecmap * mask

            if len(img.shape) == 3:
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)
                vecmap = np.swapaxes(vecmap, 1, 2)
                vecmap = np.swapaxes(vecmap, 1, 0)
                heatmap = np.swapaxes(heatmap, 1, 2)
                heatmap = np.swapaxes(heatmap, 1, 0)
    
            img_mean = np.array([128.0, 128.0, 128.0])[:, np.newaxis, np.newaxis].astype('float32')
            img_std = np.array([256.0, 256.0, 256.0])[:, np.newaxis, np.newaxis].astype('float32')
            
            img -= img_mean
            img /= img_std
            
            vecmask = np.repeat(mask, len(self.vec_pair[0]) * 2)
            heatmask = np.repeat(mask, len(self.vec_pair[0]))

            yield img.astype('float32'), vecmap.astype('float32'), heatmap.astype('float32'), vecmask.astype('float32'), heatmask.astype('float32')

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    traindir = ['/home/xingzhaolong/dataset/coco/filelist/train_list.txt'  ,'/home/xingzhaolong/dataset/coco/masklist/train_mask_list.txt', '/home/xingzhaolong/dataset/coco/json/train.json']
    filereader = CocoFolder(traindir, 8,
            Mytransforms.Compose([Mytransforms.RandomResized(),
            Mytransforms.RandomRotate(40),
            Mytransforms.RandomCrop(368),
            Mytransforms.RandomHorizontalFlip(),
        ])).reader

    index = 14500
    for file in filereader():
        print index 
        index += 1
    pass
