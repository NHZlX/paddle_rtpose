### generated by caffe2fluid, your net is in class "" ###

import math
import os
import numpy as np
from paddle.v2.fluid.initializer import Xavier


def import_fluid():
    import paddle.v2.fluid as fluid
    return fluid


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.paddle_env = None
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, exe=None, place=None, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        fluid = import_fluid()
        #load fluid mode directly
        if os.path.isdir(data_path):
            assert (exe is not None), \
                'must provide a executor to load fluid model'
            fluid.io.load_persistables_if_exist(executor=exe, dirname=data_path)
            return True

        #load model from a npy file
        if exe is None or place is None:
            if self.paddle_env is None:
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                self.paddle_env = {'place': place, 'exe': exe}
                exe = exe.run(fluid.default_startup_program())
            else:
                place = self.paddle_env['place']
                exe = self.paddle_env['exe']

        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            if op_name not in self.layers.keys():
                continue
            layer = self.layers[op_name]
            for param_name, data in data_dict[op_name].iteritems():
                try:
                    name = '%s_%s' % (op_name, param_name)
                    v = fluid.global_scope().find_var(name)
                    w = v.get_tensor()
                    w.set(data, place)
                except ValueError:
                    if not ignore_missing:
                        raise
        return True

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=None,
             group=1,
             biased=True):
        if padding is None:
            padding = [0, 0]

        # Get the number of channels in the input
        c_i, h_i, w_i = input.shape[1:]

        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        fluid = import_fluid()
        prefix = name + '_'
        output = fluid.layers.conv2d(
            input=input,
            filter_size=[k_h, k_w],
            num_filters=c_o,
            stride=[s_h, s_w],
            padding=padding,
            groups=group,
            param_attr=fluid.ParamAttr(name=prefix + "weights", 
                                      initializer=Xavier()),
            bias_attr=fluid.ParamAttr(name=prefix + "biases"),
            act="relu" if relu is True else None)
        return output

    @layer
    def relu(self, input, name):
        fluid = import_fluid()
        output = fluid.layers.relu(x=input)
        return output

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=None):
        if padding is None:
            padding = [0, 0]

        # Get the number of channels in the input
        h_i, w_i = input.shape[2:]
        fluid = import_fluid()
        output = fluid.layers.pool2d(
            input=input,
            pool_size=[k_h, k_w],
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            pool_type='max')
        return output

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=None):
        if padding is None:
            padding = [0, 0]

        # Get the number of channels in the input
        h_i, w_i = input.shape[2:]
        fluid = import_fluid()
        output = fluid.layers.pool2d(
            input=input,
            pool_size=[k_h, k_w],
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            pool_type='avg')
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        raise Exception('lrn() not implemented yet')

    @layer
    def concat(self, inputs, axis, name):
        fluid = import_fluid()
        output = fluid.layers.concat(input=inputs, axis=axis)
        return output

    @layer
    def add(self, inputs, name):
        fluid = import_fluid()
        output = inputs[0]
        for i in inputs[1:]:
            output = fluid.layers.elementwise_add(x=output, y=i)
        return output

    @layer
    def fc(self, input, num_out, name, relu=True, act=None):
        fluid = import_fluid()

        if act is None:
            act = 'relu' if relu is True else None

        prefix = name + '_'
        output = fluid.layers.fc(
            name=name,
            input=input,
            size=num_out,
            act=act,
            param_attr=fluid.ParamAttr(name=prefix + 'weights'),
            bias_attr=fluid.ParamAttr(name=prefix + 'biases'))
        return output

    @layer
    def softmax(self, input, name):
        fluid = import_fluid()
        output = fluid.layers.softmax(x=input, name=name)
        return output

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        fluid = import_fluid()
        prefix = name + '_'
        param_attr = None if scale_offset is False else fluid.ParamAttr(
            name=prefix + 'scale')
        bias_attr = None if scale_offset is False else fluid.ParamAttr(
            name=prefix + 'offset')
        mean_name = prefix + 'mean'
        variance_name = prefix + 'variance'
        output = fluid.layers.batch_norm(
            name=name,
            input=input,
            is_test=True,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=mean_name,
            moving_variance_name=variance_name,
            epsilon=1e-5,
            act='relu' if relu is True else None)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        raise Exception('dropout() not implemented yet')


class Pose(Network):
    def setup(self):
        self.feed('data')
        self.conv(3, 3, 64, 1, 1, padding=[1, 1], name='conv1_1')
        self.conv(3, 3, 64, 1, 1, padding=[1, 1], name='conv1_2')
        self.max_pool(2, 2, 2, 2, name='pool1_stage1')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv2_1')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv2_2')
        self.max_pool(2, 2, 2, 2, name='pool2_stage1')
        self.conv(3, 3, 256, 1, 1, padding=[1, 1], name='conv3_1')
        self.conv(3, 3, 256, 1, 1, padding=[1, 1], name='conv3_2')
        self.conv(3, 3, 256, 1, 1, padding=[1, 1], name='conv3_3')
        self.conv(3, 3, 256, 1, 1, padding=[1, 1], name='conv3_4')
        self.max_pool(2, 2, 2, 2, name='pool3_stage1')
        self.conv(3, 3, 512, 1, 1, padding=[1, 1], name='conv4_1')
        self.conv(3, 3, 512, 1, 1, padding=[1, 1], name='conv4_2')
        self.conv(3, 3, 256, 1, 1, padding=[1, 1], name='conv4_3_CPM')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv4_4_CPM')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_1_CPM_L1')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_2_CPM_L1')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_3_CPM_L1')
        self.conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='conv5_5_CPM_L1')

        self.feed('conv4_4_CPM')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_1_CPM_L2')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_2_CPM_L2')
        self.conv(3, 3, 128, 1, 1, padding=[1, 1], name='conv5_3_CPM_L2')
        self.conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='conv5_5_CPM_L2')

        self.feed('conv5_5_CPM_L1', 
                  'conv5_5_CPM_L2', 
                  'conv4_4_CPM')
        self.concat(1, name='concat_stage2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage2_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage2_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage2_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage2_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage2_L1')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage2_L1')

        self.feed('concat_stage2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage2_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage2_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage2_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage2_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage2_L2')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage2_L2')

        self.feed('Mconv7_stage2_L1', 
                  'Mconv7_stage2_L2', 
                  'conv4_4_CPM')
        self.concat(1, name='concat_stage3')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage3_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage3_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage3_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage3_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage3_L1')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage3_L1')

        self.feed('concat_stage3')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage3_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage3_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage3_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage3_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage3_L2')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage3_L2')

        self.feed('Mconv7_stage3_L1', 
                  'Mconv7_stage3_L2', 
                  'conv4_4_CPM')
        self.concat(1, name='concat_stage4')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage4_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage4_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage4_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage4_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage4_L1')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage4_L1')

        self.feed('concat_stage4')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage4_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage4_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage4_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage4_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage4_L2')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage4_L2')

        self.feed('Mconv7_stage4_L1', 
                  'Mconv7_stage4_L2', 
                  'conv4_4_CPM')
        self.concat(1, name='concat_stage5')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage5_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage5_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage5_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage5_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage5_L1')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage5_L1')

        self.feed('concat_stage5')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage5_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage5_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage5_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage5_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage5_L2')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage5_L2')

        self.feed('Mconv7_stage5_L1', 
                  'Mconv7_stage5_L2', 
                  'conv4_4_CPM')
        self.concat(1, name='concat_stage6')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage6_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage6_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage6_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage6_L1')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage6_L1')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
        self.conv(1, 1, 38, 1, 1, relu=False, name='Mconv7_stage6_L1')

        self.feed('concat_stage6')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv1_stage6_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv2_stage6_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv3_stage6_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv4_stage6_L2')
        self.conv(7, 7, 128, 1, 1, padding=[3, 3], name='Mconv5_stage6_L2')
        self.conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
        self.conv(1, 1, 19, 1, 1, relu=False, name='Mconv7_stage6_L2')

    @classmethod
    def convert(cls, npy_model, fluid_path):
        import paddle.v2.fluid as fluid
        data_layer = fluid.layers.data(name="data", shape=[3,368,368], dtype="float32")
        feed_data = {"data": data_layer}
        net = cls(feed_data)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        net.load(data_path=npy_model, exe=exe, place=place)
        fluid.io.save_persistables(executor=exe, dirname=fluid_path)

if __name__ == "__main__":
    #usage: python xxxnet.py xxx.npy ./model

    import sys
    npy_weight = sys.argv[1]
    fluid_model = sys.argv[2]
    Pose.convert(npy_weight, fluid_model)
    exit(0)