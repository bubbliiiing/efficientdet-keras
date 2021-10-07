#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.efficientdet import efficientdet

if __name__ == "__main__":
    num_classes = 20

    model = efficientdet(0, num_classes)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
