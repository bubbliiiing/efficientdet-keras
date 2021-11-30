#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.efficientdet import efficientdet
from utils.utils import get_classes, image_sizes

if __name__ == "__main__":
    num_classes = 20
    phi         = 0
    input_shape = [image_sizes[phi], image_sizes[phi]]

    model = efficientdet(input_shape, phi, num_classes)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
