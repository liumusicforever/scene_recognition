

import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize, MinMaxScaler

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from matchers.matcher_base import MatcherBase


class PretrainedCnnMatcher(MatcherBase):
    def __init__(self):
        self.batch_size = 10
        self.model = ResNet50(weights='imagenet',
                              include_top=False, pooling='avg')
        self.dist = DistanceMetric.get_metric('euclidean')
        self.min_max_scaler = MinMaxScaler()

    def distance_metrics(self, scenes, is_norm=True):
        feats = self.get_feats(scenes)
        dist_metrics = self.dist.pairwise(feats, feats)
        print(dist_metrics)
        if is_norm:
            # dist_metrics = normalize(dist_metrics, axis=0, norm='l2')
            dist_metrics = self.min_max_scaler.fit_transform(dist_metrics)
        return dist_metrics

    def _preprocessing(self, scenes):
        batch_imgs = []
        for img_path in scenes:
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            batch_imgs.append(img)
        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_imgs = preprocess_input(batch_imgs)
        return batch_imgs

    def get_feats(self, scenes):
        batch_feats = []
        for i in range(int(len(scenes)/self.batch_size)+1):
            start = i*self.batch_size
            end = min((i+1)*self.batch_size, len(scenes))
            _scenes = scenes[start:end]
            if len(_scenes) == 0:
                break
            _batch_imgs = self._preprocessing(_scenes)
            _batch_feats = self.model.predict(_batch_imgs)
            for feat in _batch_feats:
                batch_feats.append(feat)
        return np.array(batch_feats)


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    matcher = PretrainedCnnMatcher()
    print(scenes)
    print(matcher.distance_metrics(scenes))
