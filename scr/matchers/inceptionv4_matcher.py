import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

import random

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import skimage
import skimage.io
import skimage.transform
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize, MinMaxScaler

from matchers.matcher_base import MatcherBase
from matchers.distance_metrics import DistanceBuilder

sys.path.append("/root/dennis_code_base/models/research/slim/")
from nets import inception_v4
from nets import inception_utils


class InceptionV4Matcher(MatcherBase):
    def __init__(self, ckpt_path):
        checkpoints_dir = ckpt_path
        self.batch_size = 10
        # classify initial
        inception_v4_arg_scope = inception_utils.inception_arg_scope()
        self.image_size = 299
        arg_scope = inception_utils.inception_arg_scope()
        number_classes = 1001
        self.input_batch = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3])
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v4.inception_v4(self.input_batch, num_classes=number_classes, is_training=False)

        weights_restored_from_file = slim.get_variables_to_restore(
            exclude=['InceptionV4/Logits'])

        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
                                                 weights_restored_from_file, ignore_missing_vars=True)  # assign varible

        self.feature_global = end_points['global_pool']

        self.sess = tf.Session()
        init_fn(self.sess)

        self.dist = DistanceBuilder()

    def distance_metrics(self, scenes, is_norm=True):
        feats = self.get_feats(scenes)
        dist_metrics = self.dist.over_all_distance(feats)
        return 1 - dist_metrics





    def _preprocessing(self, scenes):
        batch_imgs = []
        for img_path in scenes:
            img = self.load_image(img_path)

            if img.shape[-1] == 4:
                img = img[..., :3]

            batch_imgs.append(img)
        batch_imgs = np.array(batch_imgs)
        return batch_imgs

    def extract_feature(self, images):
        feature = self.sess.run(self.feature_global, feed_dict={self.input_batch: images})
        embedding = np.array(feature).reshape(len(images), -1)
        return embedding

    def get_feats(self, scenes):
        batch_feats = []
        for i in range(int(len(scenes) / self.batch_size) + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(scenes))
            _scenes = scenes[start:end]
            if len(_scenes) == 0:
                break
            _batch_imgs = self._preprocessing(_scenes)
            _batch_feats = self.extract_feature(_batch_imgs)
            for feat in _batch_feats:
                batch_feats.append(feat)
        return np.array(batch_feats)

    def load_image(self, path, shift=False):  # 下面的shift是True
        # load image
        img = skimage.io.imread(path)  # img is numpy array

        if shift:  # default is Flase
            angle_idx = random.choice([0, 1, 2, 3])
            img = rotate_bound(img, angle_idx * 90)

            # angle =random.uniform(0, 1) * 360
            # img = rotate_bound(img,angle)

        img = img / 255.0  # regulration
        assert (0 <= img).all() and (img <= 1.0).all()  # assert算是提醒用的
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])  # find short side
        if shift:
            x_off = random.uniform(0.90, 1.1)
            y_off = random.uniform(0.90, 1.1)
            yy = max(int((img.shape[0] - short_edge * y_off) / 2), 0)
            xx = max(int((img.shape[1] - short_edge * x_off) / 2), 0)
        else:
            yy = int((img.shape[0] - short_edge) / 2)  # row  shaindexpe
            xx = int((img.shape[1] - short_edge) / 2)  # column
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]  #
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (self.image_size, self.image_size), mode='constant')
        return resized_img

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))





if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    ckpt_path = '/root/data/scene_recognition/static//checkpoints'
    matcher = InceptionV4Matcher(ckpt_path)
    print(scenes)
    print(matcher.distance_metrics(scenes))
