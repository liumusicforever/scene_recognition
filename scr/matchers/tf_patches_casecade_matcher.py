import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

import concurrent.futures

import cv2
import numpy as np

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from keras.applications.resnet50 import ResNet50

from matchers.matcher_base import MatcherBase
from matchers.tf_cnn_matcher import TFCnnMatcher

from tqdm import tqdm


class TFPatchesCaseCadeMatcher(MatcherBase):
    def __init__(self, pb_path, delta=0.8, crop_ratio_list=[0.7,0.5,0.3], percent = 0.4):
        self.batch_size = 10
        self.matcher = TFCnnMatcher(pb_path)
        self.dist = DistanceMetric.get_metric('euclidean')
        self.min_max_scaler = MinMaxScaler()

        self.resize = 224
        self.delta = delta
        self.crop_ratio_list = crop_ratio_list
        self.percent = percent

    def distance_metrics(self, scenes, is_norm=True):

        features = []
        for img_path in tqdm(scenes):
            img = self._imread(img_path)
            patches = self._gen_anchors(img)
            feats = self.get_feats(patches)
            features.append(feats)

        dist_metrics = np.zeros((len(scenes), len(scenes)))
        for i in tqdm(range(len(features))):
            for j in range(i + 1, len(features)):
                feats1 = features[i]
                feats2 = features[j]
                dist = self.feats_distance(feats1, feats2)
                dist_metrics[i][j] = dist
                dist_metrics[j][i] = dist

        if is_norm:
            dist_metrics = (dist_metrics - np.min(dist_metrics)) / (np.max(dist_metrics) - np.min(dist_metrics))
            # dist_metrics = self.min_max_scaler.fit_transform(dist_metrics)

        if self.delta is None:
            dist_metrics = 1-dist_metrics

        return dist_metrics

    def _comput_i_j(self, features, i, j):
        feats1 = features[i]
        feats2 = features[j]
        return (i, j), self.feats_distance(feats1, feats2)

    def feats_distance(self, feats1, feats2):
        dist_metrics = self.dist.pairwise(feats1, feats2)
        if self.delta is not None:
            dist = len(dist_metrics[dist_metrics < self.delta]) / (len(feats1) * len(feats2))
        else:
            dist_metrics = np.array(dist_metrics)
            dist_metrics = dist_metrics.flatten()
            dist_metrics = np.sort(dist_metrics, axis=None)
            if self.percent > 1.0:
                dist_metrics_len = min(int(self.percent),len(dist_metrics))
            else:
                dist_metrics_len = int(len(dist_metrics) * self.percent)
                if dist_metrics_len == 0:
                    dist_metrics_len=1
            dist_metrics = dist_metrics[:dist_metrics_len]
            dist = np.average(dist_metrics)

            # dist = 1 - dist_metrics_sum

        return dist

    def get_feats(self, patches):
        batch_feats = []
        for i in range(int(len(patches) / self.batch_size) + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(patches))
            _batch_imgs = patches[start:end]

            if len(_batch_imgs) == 0:
                break
            _batch_imgs = self._preprocessing(_batch_imgs)
            _batch_imgs = np.stack(_batch_imgs, axis=0)
            _batch_feats = self.matcher.extract_feature(_batch_imgs)
            for feat in _batch_feats:
                batch_feats.append(feat)
        return np.array(batch_feats)

    def _preprocessing(self, batch_imgs):
        _batch_imgs = []
        for img in batch_imgs:
            shape = img.shape
            if shape[0] != self.resize or shape[1] != self.resize:
                img = cv2.resize(img, (self.resize, self.resize))
            _batch_imgs.append(img)
        return np.array(_batch_imgs)

    def _gen_anchors(self, img):
        def sliding_window(image, stepSize, windowSize):
            # slide a window across the image
            for y in range(0, image.shape[0], stepSize):
                for x in range(0, image.shape[1], stepSize):
                    # yield the current window
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

        # crop images
        H, W = img.shape[:2]
        segments = []
        for crop_ratio in self.crop_ratio_list:
            for x, y, patch_img in sliding_window(img, int(H * crop_ratio / 2), [int(H * crop_ratio), int(W * crop_ratio)]):
                h, w = patch_img.shape[:2]
                if min([h, w]) / max([h, w]) < 0.5:
                    continue
                segments.append(patch_img)
        return segments

    def _imread(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    SOFTMAX_MODEL_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene/softmax_resnet50/1557727162/'
    # matcher = TFPatchesMatcher(SOFTMAX_MODEL_PATH,delta=None,crop_ratio=0.4)
    # print(scenes)
    # print(matcher.distance_metrics(scenes))

    matcher = TFPatchesCaseCadeMatcher(SOFTMAX_MODEL_PATH, delta=0.85)
    print(scenes)
    print(matcher.distance_metrics(scenes))
