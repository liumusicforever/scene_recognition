import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize, MinMaxScaler

import cv2
import numpy as np
import tensorflow as tf

from matchers.matcher_base import MatcherBase


class TFCnnMatcher(MatcherBase):

    def __init__(
            self,
            pb_path,
            resize=224,
            margin_ratio=0,
            gpu_id=0):

        self.batch_size = 10
        self.dist = DistanceMetric.get_metric('euclidean')
        self.min_max_scaler = MinMaxScaler()
        self.margin_ratio = margin_ratio
        self.resize = resize
        self.gpu_id = gpu_id
        self.gpu_mem_fraction = 0.3

        with tf.device('/gpu:' + str(gpu_id)):
            self.graph = tf.Graph()
            with self.graph.as_default():
                graph_def = tf.GraphDef()

        # self.graph = tf.get_default_graph()

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=float(self.gpu_mem_fraction))

        self.sess = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(gpu_options=gpu_options))
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], pb_path)

        # get tensor
        try:
            self.images_placeholder = \
                self.graph.get_tensor_by_name("input:0")
            self.embeddings = \
                self.graph.get_tensor_by_name("embeddings:0")
        except Exception as e:
            print(e)
            print('Can not find tensor name starts with "<name>",'
                  'try "import/<name>"')
            self.images_placeholder = self.graph.get_tensor_by_name("import/input:0")
            self.embeddings = self.graph.get_tensor_by_name("import/embeddings:0")

    def distance_metrics(self, scenes, is_norm=True):
        feats = self.get_feats(scenes)
        dist_metrics = self.dist.pairwise(feats, feats)
        # print(dist_metrics)
        if is_norm:
            # dist_metrics = normalize(dist_metrics, axis=0, norm='l2')
            # dist_metrics = self.min_max_scaler.fit_transform(dist_metrics)
            dist_metrics = (dist_metrics - np.min(dist_metrics)) / (np.max(dist_metrics) - np.min(dist_metrics))
        return 1 - dist_metrics

    def _preprocessing(self, scenes):
        batch_imgs = []
        for img_path in scenes:
            bgr_img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            shape = rgb_img.shape

            if shape[0] != self.resize or shape[1] != self.resize:
                rgb_img = cv2.resize(rgb_img, (self.resize, self.resize))

            batch_imgs.append(rgb_img)
        batch_imgs = np.array(batch_imgs)
        return batch_imgs

    def extract_feature(self, images):
        """
        Args:
            images: ndarray, images of format (N, H, W, C)
        Returns
            embedding: list of feature vector
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        feed_dict = {
            self.images_placeholder: images,
        }

        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return embeddings

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


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    pb_path = '/root/data/scene_models/triplet_loss/frozen_graph/'
    matcher = TFCnnMatcher(pb_path)
    print(scenes)
    print(matcher.distance_metrics(scenes))
