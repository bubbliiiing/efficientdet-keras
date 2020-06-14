import numpy as np
import keras
from keras.optimizers import Adam
from nets.efficientdet_training import Generator
from nets.efficientdet_training import focal,smooth_l1 
from nets.efficientdet import Efficientdet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility
from utils.anchors import get_anchors

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

freeze_layers = [226, 328, 328, 373, 463, 565, 655, 802]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

if __name__ == "__main__":
    #-------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    #-------------------------------------------#
    phi = 0
    annotation_path = '2007_train.txt'

    classes_path = 'model_data/voc_classes.txt' 
    class_names = get_classes(classes_path)
    NUM_CLASSES = len(class_names)  

    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    model_path = "model_data/efficientdet-d0-voc.h5"

    model = Efficientdet(phi,num_classes=NUM_CLASSES)
    priors = get_anchors(image_sizes[phi])
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    model.load_weights(model_path,by_name=True,skip_mismatch=True)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 训练参数设置
    logging = TensorBoard(log_dir="logs")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = False

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 4
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (image_sizes[phi], image_sizes[phi]),NUM_CLASSES)
        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        model.fit_generator(
                gen.generate(True), 
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=gen.generate(False),
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Freeze_Epoch, 
                verbose=1,
                initial_epoch=Init_Epoch ,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )

    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        BATCH_SIZE = 4
        Lr = 5e-5
        Freeze_Epoch = 50
        Epoch = 100
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                        (image_sizes[phi], image_sizes[phi]),NUM_CLASSES)

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': focal()
                },optimizer=keras.optimizers.Adam(Lr)
        )   
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, BATCH_SIZE))
        model.fit_generator(
                gen.generate(True), 
                steps_per_epoch=max(1, num_train//BATCH_SIZE),
                validation_data=gen.generate(False),
                validation_steps=max(1, num_val//BATCH_SIZE),
                epochs=Epoch, 
                verbose=1,
                initial_epoch=Freeze_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )
