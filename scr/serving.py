import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), './'))

import tensorflow as tf
from matchers.pretrained_cnn_matcher import PretrainedCnnMatcher


class SceneRecognitionServing(object):
    def __init__(self):
        self.matcher = PretrainedCnnMatcher()

    def load_imgs(self):
        pass

    def distance_metrics(self, scenes):
        dist_metrics = self.matcher.distance_metrics(scenes)
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
            1-input_op,
            k=3,
            sorted=True,
            name='top_k'
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            values, indexes = sess.run(
                top_k_op, feed_dict={input_op: dist_metrics})
        return values, indexes


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

    # create scene recognition serving
    src_serving = SceneRecognitionServing()

    # Get distance metrics
    dist_metrics = src_serving.distance_metrics(scenes)

    # Split similar and not similar scene
    similar_indexes = src_serving.filter(dist_metrics)
    print(similar_indexes)
