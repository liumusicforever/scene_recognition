import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

import concurrent.futures

import cv2
import numpy as np

import tensorflow as tf

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from keras.applications.resnet50 import ResNet50

from matchers.matcher_base import MatcherBase

from tqdm import tqdm


class CnnPatchesMatcher(MatcherBase):
    def __init__(self, delta=None, crop_ratio=0.3):
        self.batch_size = 20
        self.model = ResNet50(weights='imagenet',
                              include_top=False, pooling='avg')

        self.dist = DistanceMetric.get_metric('euclidean')
        self.min_max_scaler = MinMaxScaler()

        self.delta = delta
        self.crop_ratio = crop_ratio

    def distance_metrics(self, scenes, is_norm=True):

        features = []
        for img_path in tqdm(scenes):
            img = self._imread(img_path)
            patches = self._gen_anchors(img)
            feats = self.get_feats(patches)
            features.append(feats)

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for i in tqdm(range(len(features))):
                for j in range(i + 1, len(features)):
                    results.append(executor.submit(self._comput_i_j, *(features, i, j)))

            dist_metrics = np.zeros((len(scenes), len(scenes)))
            for res in results:
                (i, j), dist = res.result()
                dist_metrics[i][j] = dist
                dist_metrics[j][i] = dist

        # dist_metrics = np.zeros((len(scenes), len(scenes)))
        # for i in tqdm(range(len(features))):
        #     for j in range(i + 1, len(features)):
        #         feats1 = features[i]
        #         feats2 = features[j]
        #         dist = self.feats_distance(feats1, feats2)
        #         dist_metrics[i][j] = dist
        #         dist_metrics[j][i] = dist

        if is_norm:
            dist_metrics = (dist_metrics - np.min(dist_metrics)) / (np.max(dist_metrics) - np.min(dist_metrics))
            # dist_metrics = self.min_max_scaler.fit_transform(dist_metrics)

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
            img_a_matched = np.amin(dist_metrics, axis=1)
            img_b_matched = np.amin(dist_metrics, axis=0)
            patches_matched = np.append(img_a_matched, img_b_matched)

            dist = 1 - np.average(patches_matched)

        return dist

    def get_feats(self, patches):
        batch_feats = []
        for i in range(int(len(patches) / self.batch_size) + 1):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(patches))
            _batch_imgs = patches[start:end]

            if len(_batch_imgs) == 0:
                break
            _batch_imgs = np.stack(_batch_imgs, axis=0)
            _batch_feats = self.model.predict(_batch_imgs)
            for feat in _batch_feats:
                batch_feats.append(feat)
        return np.array(batch_feats)

    def _gen_anchors(self, img):

        def sliding_window(image, stepSize, windowSize):
            # slide a window across the image
            for y in range(0, image.shape[0], stepSize):
                for x in range(0, image.shape[1], stepSize):
                    # yield the current window
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

        crop_ratio = self.crop_ratio
        # crop images
        H, W = img.shape[:2]
        segments = []
        for x, y, patch_img in sliding_window(img, int(H * crop_ratio / 2), [int(H * crop_ratio), int(W * crop_ratio)]):
            h, w = patch_img.shape[:2]
            if min([h, w]) / max([h, w]) < 0.5:
                continue
            patch_img = cv2.resize(patch_img, (224, 224))
            segments.append(patch_img)
        return segments

    def _imread(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1].astype(np.float32)
        return img


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    matcher = CnnPatchesMatcher()
    print(scenes)
    print(matcher.distance_metrics(scenes))
