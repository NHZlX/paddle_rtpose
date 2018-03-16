#!/usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict

cfg = edict()

# data config

cfg.TRAIN_DATA_PATH='/home/xingzhaolong/dataset/coco/images/train2017'
cfg.TRAIN_ANNO_PATH='/home/xingzhaolong/dataset/coco/annotations/person_keypoints_train2017.json'
cfg.TRAIN_IMAGELIST_FILE='/home/xingzhaolong/dataset/coco/filelist/train_list.txt'
cfg.TRAIN_MASKLIST_FILE='/home/xingzhaolong/dataset/coco/masklist/train_mask_list.txt'
cfg.TRAIN_KPTJSON_FILE='/home/xingzhaolong/dataset/coco/json/train.json'
cfg.TRAIN_MASK_PATH='/home/xingzhaolong/dataset/coco/mask/'

cfg.TEST_DATA_PATH='/home/xingzhaolong/dataset/coco/images/val2017/'
cfg.TEST_ANNO_PATH='/home/xingzhaolong/dataset/coco/annotations/person_keypoints_val2017.json'
cfg.TEST_IMAGELIST_FILE='/home/xingzhaolong/dataset/coco/filelist/test_list.txt'
cfg.TEST_MASKLIST_FILE='/home/xingzhaolong/dataset/coco/masklist/test_mask_list.txt'
cfg.TEST_KPTJSON_FILE='/home/xingzhaolong/dataset/coco/json/test.json'
cfg.TEST_MASK_PATH='/home/xingzhaolong/dataset/coco/mask/'

# data input process
cfg.INPUT_SIZE = 368
cfg.RANDOM_ROTATE_ANGLE=40

# training settings
cfg.LEARNING_RATE = 0.000040
cfg.MOMENTUM=0.9
cfg.WEIGHT_DECAY=0.005
cfg.BATCH_SIZE=7
cfg.NUM_PASSES=12
cfg.BATCH_SECTION = [136106, 272212, 408309]
cfg.LEARNING_RATE_SECTION=[0.000040, 0.0000132, 0.0000044, 0.000001452]

# output settings
cfg.VEC_NUM = 19 * 2
cfg.HEATMAP_NUM = 19
cfg.STRIDE = 8
