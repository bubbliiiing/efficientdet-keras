import colorsys
import os
import pickle

import keras
import numpy as np
from keras import backend as K
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont

from nets.efficientdet import Efficientdet
from utils.anchors import get_anchors
from utils.utils import (BBoxUtility, efficientdet_correct_boxes,
                         letterbox_image)

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.iou指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.iou，那么该低分框将会被剔除。

可能有些同学知道有0.5和0.5:0.95的mAP，这里的self.iou=0.5不代表mAP0.5。
如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''
def preprocess_input(image):
    image /= 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    return image

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和phi都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class EfficientDet(object):
    _defaults = {
        "model_path"    : 'model_data/efficientdet-d0-voc.h5',
        "classes_path"  : 'model_data/voc_classes.txt',
        "phi"           : 0,
        "confidence"    : 0.4,
        "iou"           : 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化efficientdet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.model_image_size = [image_sizes[self.phi], image_sizes[self.phi],3]
        self.sess = K.get_session()
        self.generate()
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.iou)
        self.prior = self._get_prior()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得先验框
    #---------------------------------------------------#
    def _get_prior(self):
        data = get_anchors(image_sizes[self.phi])
        return data

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        #----------------------------------------#
        #   计算种类数量
        #----------------------------------------#
        self.num_classes = len(self.class_names)

        #----------------------------------------#
        #   创建Efficientdet模型
        #----------------------------------------#
        self.Efficientdet = Efficientdet(self.phi,self.num_classes)
        self.Efficientdet.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.model_image_size[0],self.model_image_size[1]])
        photo = np.array(crop_img,dtype = np.float32)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        photo = np.reshape(preprocess_input(photo),[1, self.model_image_size[0], self.model_image_size[1], self.model_image_size[2]])

        preds = self.Efficientdet.predict(photo)
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results = self.bbox_util.detection_out(preds, self.prior, confidence_threshold=self.confidence)
        
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results[0])<=0:
            return image
        results = np.array(results)
        
        det_label = results[0][:, 5]
        det_conf = results[0][:, 4]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 0], results[0][:, 1], results[0][:, 2], results[0][:, 3]
        #-----------------------------------------------------------#
        #   筛选出其中得分高于confidence的框 
        #-----------------------------------------------------------#
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        #-----------------------------------------------------------#
        #   去掉灰条部分
        #-----------------------------------------------------------#
        boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
