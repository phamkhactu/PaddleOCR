from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral

import uuid
import random
import math
import numpy as np
import os
import copy
import logging
import cv2


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x

class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info

class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info

class PadStride(object):
    """ padding image for model with FPN, instead PadBatch(pad_to_stride) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return im, im_info
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im, im_info



class TablePreprocess():
    def __init__(self, preprocess_list=[
            {'interp': 2, 'keep_ratio': False, 'target_size': [800, 608], 'type': 'Resize'},
            {'is_scale': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'type': 'NormalizeImage'},
            {'type': 'Permute'},
            {'stride': 32, 'type': 'PadStride'}
        ]) -> None:
        self.preprocess_ops = self.create_ops_preprocess(preprocess_list)
        
    def create_ops_preprocess(self, preprocess_list):
        preprocess_ops = []
        for op_info in preprocess_list:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops
    
    def __call__(self, images:list):
        if not isinstance(images, list):
            images = [images]
            
        input_im_lst = []
        input_im_info_lst = []
        
        input_im_lst = []
        input_im_info_lst = []
        for im in images:
            im, im_info = self.preprocess(im, self.preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = self.create_inputs(input_im_lst, input_im_info_lst)
        
        return inputs
    
    def create_inputs(self, imgs, im_info):
        """generate input for different model type
        Args:
            imgs (list(numpy)): list of images (np.ndarray)
            im_info (list(dict)): list of image info
        Returns:
            inputs (dict): input of model
        """
        inputs = {}

        im_shape = []
        scale_factor = []
        if len(imgs) == 1:
            inputs['image'] = np.array((imgs[0], )).astype('float32')
            inputs['im_shape'] = np.array(
                (im_info[0]['im_shape'], )).astype('float32')
            inputs['scale_factor'] = np.array(
                (im_info[0]['scale_factor'], )).astype('float32')
            return inputs

        for e in im_info:
            im_shape.append(np.array((e['im_shape'], )).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

        inputs['im_shape'] = np.concatenate(im_shape, axis=0)
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = img
            padding_imgs.append(padding_im)
        inputs['image'] = np.stack(padding_imgs, axis=0)
        return inputs

    
    def decode_image(self, im, im_info):
        """read rgb image
        Args:
            im_file (str|np.ndarray): input can be image path or np.ndarray
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
       
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return im, im_info

    def preprocess(self, im, preprocess_ops):
        # process image by preprocess_ops
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        im, im_info = self.decode_image(im, im_info)
        for operator in preprocess_ops:
            im, im_info = operator(im, im_info)
        return im, im_info


if __name__ == '__main__':
    table_pre = TablePreprocess()
    img = cv2.imread("docs/images/layout.jpg")
    inputs  = table_pre(img)
    print(inputs)