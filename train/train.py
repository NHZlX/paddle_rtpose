#!/bin/env python

import sys
import os 

sys.path.append('..')
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.learning_rate_decay as lr_decay

from datasets import CocoFolder
from datasets import Mytransforms
from config import cfg

from models.pose_vgg19 import Pose as MyNet  
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_gpu', default=1, type=int,
                        help='if use the gpu, 0 represents false, 1 represents true')
    parser.add_argument('--pretrained_model', default=None, type=str,
                        help='the path of the pretained model')
    return parser.parse_args()


def get_mask_loss(input, label, mask):
    input_mask = input * mask
    loss = fluid.layers.square_error_cost(input=input_mask, label=label)
    return loss

def main(args):
    """ main
    """
    model_path = args.pretrained_model

    paddle.init(use_gpu=args.with_gpu)

    #1, define network topology
    input_size = cfg.INPUT_SIZE
    output_size = cfg.INPUT_SIZE / cfg.STRIDE

    image = fluid.layers.data(name='image',
                              shape=[3, input_size, input_size], dtype='float32')
    vecmap = fluid.layers.data(name='vecmap',
                               shape=[cfg.VEC_NUM, output_size, output_size], dtype='float32')
    heatmap = fluid.layers.data(name='heatmap',
                                shape=[cfg.HEATMAP_NUM, output_size, output_size], dtype='float32')
    vecmask = fluid.layers.data(name='vecmask',
                                shape=[cfg.VEC_NUM, output_size, output_size], dtype='float32')
    heatmask = fluid.layers.data(name='heatmask',
                                 shape=[cfg.HEATMAP_NUM, output_size, output_size], dtype='float32')

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

    loss1_1 = get_mask_loss(vec1, vecmap, vecmask)
    loss1_2 = get_mask_loss(heatmap1, heatmap, heatmask)
    loss2_1 = get_mask_loss(vec2, vecmap, vecmask)
    loss2_2 = get_mask_loss(heatmap2, heatmap, heatmask)
    loss3_1 = get_mask_loss(vec3, vecmap, vecmask)
    loss3_2 = get_mask_loss(heatmap3, heatmap, heatmask)
    loss4_1 = get_mask_loss(vec4, vecmap, vecmask)
    loss4_2 = get_mask_loss(heatmap4, heatmap, heatmask)
    loss5_1 = get_mask_loss(vec5, vecmap, vecmask)
    loss5_2 = get_mask_loss(heatmap5, heatmap, heatmask)
    loss6_1 = get_mask_loss(vec6, vecmap, vecmask)
    loss6_2 = get_mask_loss(heatmap6, heatmap, heatmask)

    
    loss1 = loss1_1 + loss2_1 + loss3_1 + loss4_1 + loss5_1 + loss6_1
    cost1 = fluid.layers.mean(x=loss1)

    loss2 = loss1_2 + loss2_2 + loss3_2 + loss4_2+ loss5_2 + loss6_2
    cost2 = fluid.layers.mean(x=loss2)
   
    avg_cost = cost1 + cost2
    #avg_cost = fluid.layers.mean(x=cost)

    model_save_dir = '../models/checkpoints'

    global_step = layers.create_global_var(
                     shape=[1], value=0.0, dtype='float32', persistable=True, force_cpu=True)
    lr_rate = lr_decay.piecewise_decay(global_step, values=cfg.LEARNING_RATE_SECTION, boundaries = cfg.BATCH_SECTION)

    # set learning_rate batch_size  num_passes  model_save_dir

    optimizer = fluid.optimizer.Momentum(
        learning_rate=cfg.LEARNING_RATE,
        global_step=global_step,
        momentum=cfg.MOMENTUM,
        regularization=fluid.regularizer.L2Decay(cfg.WEIGHT_DECAY))

    opts = optimizer.minimize(avg_cost)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = [avg_cost] 
        inference_program = fluid.io.get_inference_program(test_target)

    place = fluid.CUDAPlace(3) if args.with_gpu is True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader =  paddle.batch(
        CocoFolder.CocoFolder(
                cfg.TRAIN_DATA_PATH,
                [cfg.TRAIN_IMAGELIST_FILE,
                 cfg.TRAIN_MASKLIST_FILE,
                 cfg.TRAIN_KPTJSON_FILE],
                cfg.STRIDE,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(cfg.RANDOM_ROTATE_ANGLE),
                Mytransforms.RandomCrop(cfg.INPUT_SIZE),
                Mytransforms.RandomHorizontalFlip(),
            ])).reader,
        batch_size=cfg.BATCH_SIZE
    )

    feeder = fluid.DataFeeder(place=place, feed_list=[image, vecmap, heatmap, vecmask, heatmask])

    if not model_path:
        pass
    elif model_path.find('.npy') > 0:
        net.load(data_path=model_path, exe=exe, place=place)
    else:
        net.load(data_path=model_path, exe=exe)

    for pass_id in range(cfg.NUM_PASSES):
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
        test loss needed
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
    main(args)
