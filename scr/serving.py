import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), './'))

import tensorflow as tf
from matchers.pretrained_cnn_matcher import PretrainedCnnMatcher
from matchers.cnn_patches_matcher import CnnPatchesMatcher
from matchers.tf_patches_matcher import TFPatchesMatcher
from matchers.tf_cnn_matcher import TFCnnMatcher

TRIPLET_MODEL_PATH = '/root/data/scene_models/triplet_loss/frozen_graph/'
SOFTMAX_MODEL_PATH = '/root/data/scene_models/softmax_loss/frozen_graph/'


class SceneRecognitionServing(object):
    def __init__(self, model='cnn', k_similar=3):
        if model == 'cnn':
            self.matcher = PretrainedCnnMatcher()
        elif model == 'patch_cnn':
            self.matcher = CnnPatchesMatcher()
        elif model == 'triplet_cnn':
            pb_path = TRIPLET_MODEL_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_cnn':
            pb_path = SOFTMAX_MODEL_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_patch_cnn':
            pb_path = SOFTMAX_MODEL_PATH
            self.matcher = TFPatchesMatcher(pb_path)
        elif model == 'triplet_patch_cnn':
            pb_path = TRIPLET_MODEL_PATH
            self.matcher = TFPatchesMatcher(pb_path)
        else:
            assert 'Not support Matcher'
        self.k_similar = k_similar

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def load_imgs(self):
        pass

    def distance_metrics(self, scenes, is_norm=True):
        dist_metrics = self.matcher.distance_metrics(scenes, is_norm)
        return dist_metrics

    def filter(self, dist_metrics, thresh=0.3):
        similar_indexes = []
        values, indexes = self.get_top_k(dist_metrics)
        for base_index, (k_values, k_indexes) in enumerate(zip(values, indexes)):
            if all(v >= thresh for v in k_values):
                similar_indexes.append(base_index)
            # for value, compare_index in zip(k_values, k_indexes):
            #     if base_index == compare_index:
            #         continue
            #     if value >= 0.3:
            #         pass

        return similar_indexes

    def get_top_k(self, dist_metrics):
        input_op = tf.placeholder(dtype=tf.float64)
        top_k_op = tf.nn.top_k(
            input_op,
            k=self.k_similar,
            sorted=True,
            name='top_k'
        )

        values, indexes = self.sess.run(
            top_k_op, feed_dict={input_op: dist_metrics})
        return values, indexes


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    # create scene recognition serving
    src_serving = SceneRecognitionServing(model='softmax_patch_cnn')

    # Get distance metrics
    dist_metrics = src_serving.distance_metrics(scenes, is_norm=True)

    # Split similar and not similar scene
    similar_indexes = src_serving.filter(dist_metrics, 0.1)
    print(dist_metrics)
    print(similar_indexes)
