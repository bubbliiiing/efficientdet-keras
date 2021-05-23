import keras
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)

from nets.efficientdet import Efficientdet
from nets.efficientdet_training import Generator, focal, smooth_l1, LossHistory
from utils.anchors import get_anchors
from utils.utils import BBoxUtility


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

freeze_layers   = [226, 328, 328, 373, 463, 565, 655, 802]
image_sizes     = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    #-------------------------------------------#
    phi = 0
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------#
    #   classes的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt' 
    #------------------------------------------------------#
    #   一共有多少类和多少先验框
    #------------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)  
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = "model_data/efficientdet-d0-voc.h5"

    #------------------------------------------------------#
    #   创建Efficientdet模型
    #------------------------------------------------------#
    model = Efficientdet(phi,num_classes=num_classes)
    model.load_weights(model_path,by_name=True,skip_mismatch=True)
    
    #-------------------------------#
    #   获得先验框
    #-------------------------------#
    priors = get_anchors(image_sizes[phi])
    bbox_util = BBoxUtility(num_classes, priors)

    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir="logs")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory("logs")


    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = False

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size      = 8
        Lr              = 1e-3
        Init_Epoch      = 0
        Freeze_Epoch    = 50

        gen             = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (image_sizes[phi], image_sizes[phi]), num_classes)

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        model.fit_generator(
                gen.generate(True), 
                steps_per_epoch=epoch_size,
                validation_data=gen.generate(False),
                validation_steps=epoch_size_val,
                epochs=Freeze_Epoch, 
                verbose=1,
                initial_epoch=Init_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )

    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size      = 4
        Lr              = 5e-5
        Freeze_Epoch    = 50
        Epoch           = 100
        
        gen             = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                                        (image_sizes[phi], image_sizes[phi]), num_classes)

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   
        
        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        model.fit_generator(
                gen.generate(True), 
                steps_per_epoch=epoch_size,
                validation_data=gen.generate(False),
                validation_steps=epoch_size_val,
                epochs=Epoch, 
                verbose=1,
                initial_epoch=Freeze_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
            )
