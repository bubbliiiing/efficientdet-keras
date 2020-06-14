import numpy as np
import keras
import pickle
import matplotlib.pyplot as plt

def decode_boxes(mbox_loc, mbox_priorbox):
    # 获得先验框的宽与高
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    # 获得先验框的中心点
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height
    decode_bbox_center_y += prior_center_y
    
    # 真实框的宽与高的求取
    decode_bbox_width = np.exp(mbox_loc[:, 2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3])
    decode_bbox_height *= prior_height

    # 获取真实框的左上角与右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    # 真实框的左上角与右下角进行堆叠
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                    decode_bbox_ymin[:, None],
                                    decode_bbox_xmax[:, None],
                                    decode_bbox_ymax[:, None]), axis=-1)

    return decode_bbox

class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    print(anchors)
    return anchors

def shift(shape, stride, anchors):
    # [0-64]
    # [0.5-64.5]
    shift_x = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    # print(shifted_anchors)
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    
    if shape[0]==4:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.ylim(-300,900)
        plt.xlim(-600,600)

        plt.scatter(shift_x,shift_y)
        box_widths = shifted_anchors[:,2]-shifted_anchors[:,0]
        box_heights = shifted_anchors[:,3]-shifted_anchors[:,1]
        
        for i in [108,109,110,111,112,113,114,115,116]:
            rect = plt.Rectangle([shifted_anchors[i, 0],shifted_anchors[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
            ax.add_patch(rect)
        plt.gca().invert_yaxis()

        ax = fig.add_subplot(122)
        plt.ylim(-300,900)
        plt.xlim(-600,600)
        plt.scatter(shift_x,shift_y)
        P7_num_anchors = len(shifted_anchors)
        random_inputs = np.random.uniform(0,1,[P7_num_anchors,4])/10
        after_decode = decode_boxes(random_inputs, shifted_anchors)
        
        box_widths = after_decode[:,2]-after_decode[:,0]
        box_heights = after_decode[:,3]-after_decode[:,1]

        after_decode_center_x = after_decode[:,0]/2+after_decode[:,2]/2
        after_decode_center_y = after_decode[:,1]/2+after_decode[:,3]/2
        plt.scatter(after_decode_center_x[108:116],after_decode_center_y[108:116])

        for i in [108,109,110,111,112,113,114,115,116]:
            rect = plt.Rectangle([after_decode[i, 0],after_decode[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
            ax.add_patch(rect)
        plt.gca().invert_yaxis()
        
        plt.show()



    return shifted_anchors

border = 512
shape = [border,border]

a = [64,32,16,8,4]

all_anchors = []
for i in range(5):
    anchors = generate_anchors(AnchorParameters.default.sizes[i])
    shifted_anchors = shift([a[i],a[i]], AnchorParameters.default.strides[i], anchors)
    all_anchors.append(shifted_anchors)

all_anchors = np.concatenate(all_anchors,axis=0)
all_anchors = all_anchors/border
all_anchors = all_anchors.clip(0,1)

print(all_anchors)