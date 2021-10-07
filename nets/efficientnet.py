import collections
import math
import string

from keras import backend, layers

MOMENTUM = 0.99
EPSILON = 1e-3

#-------------------------------------------------#
#   一共七个大结构块，每个大结构块都有特定的参数
#-------------------------------------------------#
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

#-------------------------------------------------#
#   Kernel的初始化器
#-------------------------------------------------#
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

#-------------------------------------------------#
#   Swish激活函数
#-------------------------------------------------#
def get_swish():
    def swish(x):
        return x * backend.sigmoid(x)
    return swish

#-------------------------------------------------#
#   Dropout层
#-------------------------------------------------#
def get_dropout():
    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout

#-------------------------------------------------#
#   该函数的目的是保证filter的大小可以被8整除
#-------------------------------------------------#
def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

#-------------------------------------------------#
#   计算模块的重复次数
#-------------------------------------------------#
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix=''):
    Dropout = get_dropout()

    #-------------------------------------------------#
    #   利用Inverted residuals
    #   part1 利用1x1卷积进行通道数上升
    #-------------------------------------------------#
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    #------------------------------------------------------#
    #   如果步长为2x2的话，利用深度可分离卷积进行高宽压缩
    #   part2 利用3x3卷积对每一个channel进行卷积
    #------------------------------------------------------#
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    #------------------------------------------------------#
    #   完成深度可分离卷积后
    #   对深度可分离卷积的结果施加注意力机制
    #------------------------------------------------------#
    if 0 < block_args.se_ratio <= 1:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
        se_tensor = layers.Reshape((1, 1, filters), name=prefix + 'se_reshape')(se_tensor)
        #------------------------------------------------------#
        #   通道先压缩后上升，最后利用sigmoid将值固定到0-1之间
        #------------------------------------------------------#
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    #------------------------------------------------------#
    #   part3 利用1x1卷积进行通道下降
    #------------------------------------------------------#
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=prefix + 'project_bn')(x)

    #------------------------------------------------------#
    #   part4 如果满足残差条件，那么就增加残差边
    #------------------------------------------------------#
    if block_args.id_skip and all(s == 1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=prefix + 'drop')(x)
        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 inputs=None,
                 **kwargs):
    activation  = get_swish(**kwargs)

    img_input   = inputs
    #-------------------------------------------------#
    #   创建stem部分
    #-------------------------------------------------#
    x = img_input
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    features = []
    #-------------------------------------------------#
    #   计算总的efficient_block的数量
    #-------------------------------------------------#
    block_num = 0
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    #------------------------------------------------------------------------------#
    #   对结构块参数进行循环、一共进行7个大的结构块。
    #   每个大结构块下会重复小的efficient_block
    #------------------------------------------------------------------------------#
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        #-------------------------------------------------#
        #   对使用到的参数进行更新
        #-------------------------------------------------#
        block_args = block_args._replace(
            input_filters   = round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters  = round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat      = round_repeats(block_args.num_repeat, depth_coefficient))

        # 计算drop_rate
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            #-------------------------------------------------#
            #   对使用到的参数进行更新
            #-------------------------------------------------#
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            for bidx in range(block_args.num_repeat - 1):
                # 计算drop_rate
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = mb_conv_block(x, block_args,
                                  activation = activation,
                                  drop_rate = drop_rate,
                                  prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1]))
                block_num += 1

        if idx < len(blocks_args) - 1 and blocks_args[idx + 1].strides[0] == 2:
            features.append(x)
        elif idx == len(blocks_args) - 1:
            features.append(x)
    return features

def EfficientNetB0(inputs=None, **kwargs):
    return EfficientNet(1.0, 1.0, inputs=inputs, **kwargs)


def EfficientNetB1(inputs=None, **kwargs):
    return EfficientNet(1.0, 1.1, inputs=inputs, **kwargs)


def EfficientNetB2(inputs=None, **kwargs):
    return EfficientNet(1.1, 1.2, inputs=inputs, **kwargs)


def EfficientNetB3(inputs=None, **kwargs):
    return EfficientNet(1.2, 1.4, inputs=inputs, **kwargs)


def EfficientNetB4(inputs=None, **kwargs):
    return EfficientNet(1.4, 1.8, inputs=inputs, **kwargs)


def EfficientNetB5(inputs=None, **kwargs):
    return EfficientNet(1.6, 2.2, inputs=inputs, **kwargs)


def EfficientNetB6(inputs=None, **kwargs):
    return EfficientNet(1.8, 2.6, inputs=inputs, **kwargs)


def EfficientNetB7(inputs=None, **kwargs):
    return EfficientNet(2.0, 3.1, inputs=inputs, **kwargs)

