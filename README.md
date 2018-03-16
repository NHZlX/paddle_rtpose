## PaddlePaddle Realtime Multi-Person Pose Estimation

This is a Paddle fluid version of realtime pose estimation.   
Original project is here: [https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)   
Some of this code of this project come from here: [https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)   

Contents:     
1. Requirments   
2. Run the Demo     
3. Training the model  

### Requirments

**NOTE:** Here is the [doc](http://paddlepaddle.org/docs/develop/documentation/en/build_and_install/build_from_source_en.html) about how to install PaddlePaddle.

1. [PaddlePaddle](https://github.com/PaddlePaddle/Paddle): last verion
2. For training Pose model with VGG19, you will need a Graphics card with 12G memory.


### Run the demo
**NOTE:** Need to be done by nhzlx.

### Training the model

1. Download the Training, Validation, Testing Data of COCO.

```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
2. Extract all those zip files into one directory named COCO_PATH

```
mkdir images
unzip annotations_trainval2017.zip -d ./
unzip train2017.zip -d images
unzip test2017.zip -d images
unzip val2017.zip -d images
```

3. The current directory structure will be as follows

```
$COCO_PATH
$COCO_PATH/annotations
%COCO_PATH/images
```


4. Download COCO official toolbox

**NOTE:** this toolbox provides a structure to facilitate the processing of the coco data

```
cd $COCO_PATH
git clone https://github.com/cocodataset/cocoapi.git tools 
# add the toolbox python interface path to env. 
echo "export PATH=$COCO_PATH/tools/PythonAPI/:$PATH" >> ~/.bashrc
source ~/.bashrc 
```

5. Configuration and extracting pose data from coco


```
cd $PADDLE_RTPOSE
```
There are ten variables in the `config.py` file that need to be configured, which are：

```
cfg.TRAIN_DATA_PATH='/home/xingzhaolong/dataset/coco/images/train2017'
cfg.TRAIN_ANNO_PATH='/home/xingzhaolong/dataset/coco/annotations/person_keypoints_train2017.json'

#  The following three pathes or files are what we are going to generate
cfg.TRAIN_IMAGELIST_FILE='/home/xingzhaolong/dataset/coco/filelist/train_list.txt'
cfg.TRAIN_MASKLIST_FILE='/home/xingzhaolong/dataset/coco/masklist/train_mask_list.txt'
cfg.TRAIN_KPTJSON_FILE='/home/xingzhaolong/dataset/coco/json/train.json'

cfg.TEST_DATA_PATH='/home/xingzhaolong/dataset/coco/images/val2017/'
cfg.TEST_ANNO_PATH='/home/xingzhaolong/dataset/coco/annotations/person_keypoints_val2017.json'

# The following three pathes or files are what we are going to generate
cfg.TEST_IMAGELIST_FILE='/home/xingzhaolong/dataset/coco/filelist/test_list.txt'
cfg.TEST_MASKLIST_FILE='/home/xingzhaolong/dataset/coco/masklist/test_mask_list.txt'
cfg.TEST_KPTJSON_FILE='/home/xingzhaolong/dataset/coco/json/test.json'
```

**Note:** `DATA_PATH` represents the path of the real picture, `ANNO_PATH` offers the pose annoations of the picture, `IMAGELIST_FILE` lists the name of all the pictures, `MASKLIST_FILE` lists the path of all mask files, `KPTJSON_FILE` saves some of the information of the human body, include key points, center point etc.



```
cd $PADDLE_RTPOSE/preprocess
python generate_kpt_mask.py
```

**Note:** The script will generate `IMAGELIST_FILE`, `MASKLIST_FILE`, `KPTJSON_FILE`, and this will be a period of time.



6. Download the pretained VGG19 model.

```
cd $PADDLE_RTPOSE/models
sh get_pretrained_vgg19.sh
```

7. Train the model 

```
cd $PADDLE_RTPOSE/train
sh train.sh
```
