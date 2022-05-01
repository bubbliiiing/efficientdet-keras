#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.efficientdet import efficientdet
from utils.utils import image_sizes, net_flops

if __name__ == "__main__":
    phi         = 0
    input_shape = [image_sizes[phi], image_sizes[phi], 3]
    num_classes = 80

    model = efficientdet(input_shape, phi, num_classes)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
