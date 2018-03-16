import os
import sys

import math import json
import numpy as np
from pycocotools.coco import COCO
import pickle

sys.path.insert(0,'..' )
from config import cfg

COCO_TO_OURS = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

def processing(ann_path, filelist_path, masklist_path, json_path,  mask_dir):
    coco = COCO(ann_path)
    ids = list(coco.imgs.keys())
    lists = []
    
    filelist_fp = open(filelist_path, 'w')
    masklist_fp = open(masklist_path, 'w')
    
    for i, img_id in enumerate(ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)
    
        numPeople = len(img_anns)
        name = coco.imgs[img_id]['file_name']
        height = coco.imgs[img_id]['height']
        width = coco.imgs[img_id]['width']
    
        person_centers = []
        info = dict()
        info['filename'] = name
        info['info'] = []
    
        for p in range(numPeople):
            if img_anns[p]['num_keypoints'] < 5 or img_anns[p]['area'] < 32 * 32:
                continue
            kpt = img_anns[p]['keypoints']
            dic = dict()
    
            # person center
            person_center = [img_anns[p]['bbox'][0] + img_anns[p]['bbox'][2] / 2.0, img_anns[p]['bbox'][1] + img_anns[p]['bbox'][3] / 2.0]
            scale = img_anns[p]['bbox'][3] / float(cfg.INPUT_SIZE)
    
            # skip this person if the distance to exiting person is too small
            flag = 0
            for pc in person_centers:
                dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0]) + (person_center[1] - pc[1]) * (person_center[1] - pc[1]))
                if dis < pc[2] * 0.3:
                    flag = 1;
                    break
            if flag == 1:
                continue

            dic['pos'] = person_center
            dic['keypoints'] = np.zeros((18, 3)).tolist()
            dic['scale'] = scale

            for part in range(17):
                dic['keypoints'][COCO_TO_OURS[part]][0] = kpt[part * 3]
                dic['keypoints'][COCO_TO_OURS[part]][1] = kpt[part * 3 + 1]
                # visiable is 2, unvisiable is 1 and not labeled is 0
                dic['keypoints'][COCO_TO_OURS[part]][2] = kpt[part * 3 + 2]
            
            # generate neck point based on LShoulder and RShoulder
            dic['keypoints'][1][0] = (kpt[5 * 3] + kpt[6 * 3]) * 0.5
            dic['keypoints'][1][1] = (kpt[5 * 3 + 1] + kpt[6 * 3 + 1]) * 0.5

            if kpt[5 * 3 + 2] == 0 or kpt[6 * 3 + 2] == 0:
                dic['keypoints'][1][2] = 0
            else:
                dic['keypoints'][1][2] = 1

            info['info'].append(dic)
            person_centers.append(np.append(person_center, max(img_anns[p]['bbox'][2], img_anns[p]['bbox'][3])))

        if len(info['info']) > 0:
            lists.append(info)
            filelist_fp.write(name + '\n')
            mask_all = np.zeros((height, width), dtype=np.uint8)
            mask_miss = np.zeros((height, width), dtype=np.uint8)
            flag = 0
            for p in img_anns:
                if p['iscrowd'] == 1:
                    mask_crowd = coco.annToMask(p)
                    temp = np.bitwise_and(mask_all, mask_crowd)
                    mask_crowd = mask_crowd - temp
                    flag += 1
                    continue
                else:
                    mask = coco.annToMask(p)
        
                mask_all = np.bitwise_or(mask, mask_all)
            
                if p['num_keypoints'] <= 0:
                    mask_miss = np.bitwise_or(mask, mask_miss)
        
            if flag < 1:
                mask_miss = np.logical_not(mask_miss)
            elif flag == 1:
                mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
                mask_all = np.bitwise_or(mask_all, mask_crowd)
            else:
                raise Exception('crowd segments > 1')
            
            pickle.dump(mask_miss, open(os.path.join(mask_dir, name.split('.')[0] + '.npy'), 'w'))
            masklist_fp.write(os.path.join(mask_dir, name.split('.')[0] + '.npy') + '\n')

        if i % 1000 == 0:
            print "Processed {} of {}".format(i, len(ids))
    
    masklist_fp.close()
    filelist_fp.close()
    
    fp = open(json_path, 'w')
    fp.write(json.dumps(lists))
    fp.close()
    
    print 'done!'

if __name__ == '__main__':
    processing(cfg.TRAIN_ANNO_PATH,
               cfg.TRAIN_IMAGELIST_FILE,
               cfg.TRAIN_MASKLIST_FILE,
               cfg.TRAIN_KPTJSON_FILE,
               cfg.TRAIN_MASK_PATH)

    processing(cfg.TEST_ANNO_PATH,
               cfg.TEST_IMAGELIST_FILE,
               cfg.TEST_MASKLIST_FILE,
               cfg.TEST_KPTJSON_FILE,
               cfg.TEST_MASK_PATH)
