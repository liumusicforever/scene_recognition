import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

import cv2
import numpy as np

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from keras.applications.resnet50 import ResNet50

from matchers.matcher_base import MatcherBase
from matchers.tf_cnn_matcher import TFCnnMatcher

from tqdm import tqdm


class TFPatchesMatcher(MatcherBase):
    def __init__(self, pb_path):
        self.batch_size = 10
        self.matcher = TFCnnMatcher(pb_path)
        self.dist = DistanceMetric.get_metric('euclidean')
        self.min_max_scaler = MinMaxScaler()

        self.resize = 224

    def distance_metrics(self, scenes, is_norm=True):

        features = []
        for img_path in tqdm(scenes):
            img = self._imread(img_path)
            patches = self._gen_anchors(img)
            feats = self.get_feats(patches)
            features.append(feats)

        dist_metrics = np.zeros((len(scenes), len(scenes)))
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feats1 = features[i]
                feats2 = features[j]
                dist = self.feats_distance(feats1, feats2)
                dist_metrics[i][j] = dist
                dist_metrics[j][i] = dist
        if is_norm:
            dist_metrics = (dist_metrics - np.min(dist_metrics)) / (np.max(dist_metrics) - np.min(dist_metrics))

        return dist_metrics

    def feats_distance(self, feats1, feats2, thresh=0.6):
        dist_metrics = self.dist.pairwise(feats1, feats2)
        dist = len(dist_metrics[dist_metrics < thresh]) / (len(feats1) * len(feats2))
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
            _batch_imgs = self._preprocessing(_batch_imgs)
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

    def _gen_anchors(self, img, crop_ratio=0.3):
        def sliding_window(image, stepSize, windowSize):
            # slide a window across the image
            for y in range(0, image.shape[0], stepSize):
                for x in range(0, image.shape[1], stepSize):
                    # yield the current window
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    matcher = TFPatchesMatcher()
    print(scenes)
    print(matcher.distance_metrics(scenes))
