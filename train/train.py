#!/bin/env python

#function:
#   demo to show how to use converted model using caffe2fluid
#

sys.path.append('..')
import numpy as np
import os 

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.learning_rate_decay as lr_decay

import CocoFolder
import Mytransforms

from models.pose_vgg19 import Pose as MyNet  
import argparse

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', nargs='+', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, nargs='+', type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')
    return parser.parse_args()



heatmap_weight = 46 * 46 * 19 / 2.0
vec_weight = 46 * 46 * 38 / 2.0

def get_mask_loss(input, label, mask, weight):
    input_mask = input * mask;
    #input_mask = input;
    loss = fluid.layers.square_error_cost(input=input_mask, label=label)
    #loss = fluid.layers.scale(x=loss, scale=weight)
    return loss

def main(model_path, args):
    """ main
    """
    print('load fluid model in %s' % (model_path))

    with_gpu = True
    paddle.init(use_gpu=with_gpu)

    traindir = args.train_dir
    valdir = args.val_dir

    #1, define network topology
    image = fluid.layers.data(name='image', shape=[3, 368, 368], dtype='float32')
    vecmap = fluid.layers.data(name='vecmap', shape=[38, 46, 46], dtype='float32')
    heatmap = fluid.layers.data(name='heatmap', shape=[19, 46, 46], dtype='float32')
    vecmask = fluid.layers.data(name='vecmask', shape=[38, 46, 46], dtype='float32')
    heatmask = fluid.layers.data(name='heatmask', shape=[19, 46, 46], dtype='float32')

    net = MyNet({'data': image})
    
    vec1 = net.layers['conv5_5_CPM_L1']
    heatmap1 = net.layers['conv5_5_CPM_L2']
    vec2 = net.layers['Mconv7_stage2_L1']
    heatmap2 = net.layers['Mconv7_stage2_L2']
    vec3 = net.layers['Mconv7_stage3_L1']
    heatmap3 = net.layers['Mconv7_stage3_L2']
    vec4 = net.layers['Mconv7_stage4_L1']
    heatmap4 = net.layers['Mconv7_stage4_L2']
    vec5 = net.layers['Mconv7_stage5_L1']
    heatmap5 = net.layers['Mconv7_stage5_L2']
    vec6 = net.layers['Mconv7_stage6_L1']
    heatmap6 = net.layers['Mconv7_stage6_L2']

    loss1_1 = get_mask_loss(vec1, vecmap, vecmask, vec_weight)
    loss1_2 = get_mask_loss(heatmap1, heatmap, heatmask, heatmap_weight)
    loss2_1 = get_mask_loss(vec2, vecmap, vecmask, vec_weight)
    loss2_2 = get_mask_loss(heatmap2, heatmap, heatmask, heatmap_weight)
    loss3_1 = get_mask_loss(vec3, vecmap, vecmask, vec_weight)
    loss3_2 = get_mask_loss(heatmap3, heatmap, heatmask, heatmap_weight)
    loss4_1 = get_mask_loss(vec4, vecmap, vecmask, vec_weight)
    loss4_2 = get_mask_loss(heatmap4, heatmap, heatmask, heatmap_weight)
    loss5_1 = get_mask_loss(vec5, vecmap, vecmask, vec_weight)
    loss5_2 = get_mask_loss(heatmap5, heatmap, heatmask, heatmap_weight)
    loss6_1 = get_mask_loss(vec6, vecmap, vecmask, vec_weight)
    loss6_2 = get_mask_loss(heatmap6, heatmap, heatmask, heatmap_weight)

    
    loss1 = loss1_1 + loss2_1 + loss3_1 + loss4_1 + loss5_1 + loss6_1
    cost1 = fluid.layers.mean(x=loss1)

    loss2 = loss1_2 + loss2_2 + loss3_2 + loss4_2+ loss5_2 + loss6_2
    cost2 = fluid.layers.mean(x=loss2)
   
    avg_cost = cost1 + cost2
    #avg_cost = fluid.layers.mean(x=cost)

    learning_rate = 0.000040
    batch_size = 7
    num_passes = 120
    model_save_dir = './models/'

    global_step = layers.create_global_var(
                     shape=[1], value=0.0, dtype='float32', persistable=True, force_cpu=True)
    lr_rate = lr_decay.piecewise_decay(global_step, values=[0.000040, 0.0000132, 0.0000044, 0.000001452], boundaries = [136106, 272212, 408309])

    # set learning_rate batch_size  num_passes  model_save_dir

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        global_step=global_step,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(0.0005))

    opts = optimizer.minimize(avg_cost)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = [avg_cost] 
        inference_program = fluid.io.get_inference_program(test_target)

    place = fluid.CUDAPlace(3) if with_gpu is True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader =  paddle.batch(
        CocoFolder.CocoFolder(traindir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])).reader,
        batch_size=batch_size
    )

    feeder = fluid.DataFeeder(place=place, feed_list=[image, vecmap, heatmap, vecmask, heatmask])

    #2, load weights
    if model_path.find('.npy') > 0:
        net.load(data_path=model_path, exe=exe, place=place)
    else:
        net.load(data_path=model_path, exe=exe)

    for pass_id in range(num_passes):
        for batch_id, data in enumerate(train_reader()):
            loss, step_v, lr_rate_v = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + [global_step]+ [lr_rate])
            print("Pass {0}, batch {1}, loss {2}, step {3}, lr{4}".format(
                pass_id, batch_id, loss[0], step_v[0], lr_rate_v[0]))
            if batch_id % 3000 == 0:
                model_path = os.path.join(model_save_dir, 'batch'+ str(batch_id))
                print 'save models to %s' % (model_path)
                fluid.io.save_inference_model(model_path, ['image'], [vec6, heatmap6], exe)

        '''
        for data in test_reader():
            loss = exe.run(inference_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost])
        '''

        print("End pass {0}".format(
            pass_id))

        if pass_id % 1 == 0:
            model_path = os.path.join(model_save_dir, 'pass'+ str(pass_id))
            print 'save models to %s' % (model_path)
            fluid.io.save_inference_model(model_path, ['image'], [vec6, heatmap6], exe)

if __name__ == "__main__":
    args = parse()
    main('../lenet/vgg19.npy', args)
   #main('./pose.npy', args)
