import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), './'))

import tensorflow as tf
from matchers.pretrained_cnn_matcher import PretrainedCnnMatcher
from matchers.cnn_patches_matcher import CnnPatchesMatcher
from matchers.tf_patches_matcher import TFPatchesMatcher
from matchers.tf_cnn_matcher import TFCnnMatcher
# from matchers.inceptionv4_matcher import InceptionV4Matcher
from matchers.tf_local_global_matcher import TFLocalGlobalMatcher
from matchers.tf_patches_casecade_matcher import TFPatchesCaseCadeMatcher

TRIPLET_PLACE_PATH = '/root/data/scene_models/triplet_loss/frozen_graph/'
SOFTMAX_PLACE_PATH = '/root/data/scene_models/softmax_loss/frozen_graph/'

TRIPLET_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene/triplet_resnet50/1557727201/'
SOFTMAX_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene/softmax_resnet50/1557727162/'
BASE_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene/resnet50/1559086972/'

PYRAMID_BOTTLENECK_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene0528/pyramid_bottleneck/1559148095/'
FG_ATTENTION_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene0528/fg_atten_model/1559260882/'
FG_NO_ATTENTION_GOOGLE_PATH = '/root/dennis_code_base/tf-metric-learning/experiments/scene0528/fg_no_atten_model/1559276413'


class SceneRecognitionServing(object):
    def __init__(self, model='cnn', k_similar=3):
        if model == 'pretrained':
            self.matcher = PretrainedCnnMatcher()
        if model == 'resnet50':
            pb_path = BASE_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'mle':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'mle_crop.3':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.3])
        elif model == 'mle_crop.5':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.5])
        elif model == 'mle_crop.7':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.7])
        elif model == 'mle_crop.9':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.9])

        elif model == 'mle_crop.3.5':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.3, 0.5])
        elif model == 'mle_crop.3.5.7':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.3, 0.5, 0.7])
        elif model == 'mle_crop.5.7':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.5, 0.7])
        elif model == 'mle_crop.5.7.9':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.5, 0.7, 0.9])
        elif model == 'mle_crop.7.9':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.7, 0.9])


        elif model == 'mle_delta_0.8':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=0.8, crop_ratio_list=[0.7])
        elif model == 'mle_delta_0.9':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=0.9, crop_ratio_list=[0.7])
        elif model == 'mle_delta_1.0':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.7])
        elif model == 'mle_delta_1.1':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.1, crop_ratio_list=[0.7])
        elif model == 'mle_delta_1.2':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.2, crop_ratio_list=[0.7])

        elif model == 'facenet':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)

        if model == 'cnn':
            self.matcher = PretrainedCnnMatcher()
        elif model == 'patch_cnn':
            self.matcher = CnnPatchesMatcher()
        elif model == 'triplet_place_cnn':
            pb_path = TRIPLET_PLACE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_place_cnn':
            pb_path = SOFTMAX_PLACE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_place_patch_cnn':
            pb_path = SOFTMAX_PLACE_PATH
            self.matcher = TFPatchesMatcher(pb_path)
        elif model == 'triplet_place_patch_cnn':
            pb_path = TRIPLET_PLACE_PATH
            self.matcher = TFPatchesMatcher(pb_path)
        elif model == 'base_google_cnn':
            pb_path = BASE_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'triplet_google_cnn':
            pb_path = TRIPLET_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_google_cnn':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'pyramid_bottleneck_google_cnn':
            pb_path = PYRAMID_BOTTLENECK_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'fg_attention_google_cnn':
            pb_path = FG_ATTENTION_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'fg_no_attention_google_cnn':
            pb_path = FG_NO_ATTENTION_GOOGLE_PATH
            self.matcher = TFCnnMatcher(pb_path)
        elif model == 'softmax_google_patch_cnn_60percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.6)
        elif model == 'softmax_google_patch_cnn_50percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.5)
        elif model == 'softmax_google_patch_cnn_40percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.4)
        elif model == 'softmax_google_patch_cnn_30percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.3)
        elif model == 'softmax_google_patch_cnn_20percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.2)
        elif model == 'softmax_google_patch_cnn_10percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.1)
        elif model == 'softmax_google_patch_cnn_5percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.05)
        elif model == 'softmax_google_patch_cnn_3percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.03)
        elif model == 'softmax_google_patch_cnn_1percent':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4, percent=0.01)
        elif model == 'triplet_google_patch_cnn40percent':
            pb_path = TRIPLET_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.4)
        elif model == 'softmax_google_patch_cnn_count_delta0.8':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=0.8, crop_ratio=0.4)
        elif model == 'softmax_google_patch_cnn_count_delta0.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=0.9, crop_ratio=0.4)
        elif model == 'softmax_google_patch_cnn_count_delta1.0':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.0, crop_ratio=0.4)
        elif model == 'softmax_google_patch_cnn_count_delta1.1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.1, crop_ratio=0.4)
        elif model == 'softmax_google_patch_cnn_count_delta1.2':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.2, crop_ratio=0.4)
        elif model == 'triplet_google_patch_cnn_count':
            pb_path = TRIPLET_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.1, crop_ratio=0.4)
        elif model == 'inceptionv4_cnn':
            Inception_PATH = '/root/data/scene_recognition/static//checkpoints'
            # self.matcher = InceptionV4Matcher(Inception_PATH)

        elif model == 'crop1.0percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=1.0, percent=1)
        elif model == 'crop0.95percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.95, percent=1)
        elif model == 'crop0.9percent0.1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.9, percent=0.1)

        elif model == 'crop0.9percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.9, percent=1)
        elif model == 'crop0.9percent3':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.9, percent=3)
        elif model == 'crop0.9percent5':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.9, percent=5)

        elif model == 'crop0.8percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.8, percent=1)
        elif model == 'crop0.8percent3':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.8, percent=3)
        elif model == 'crop0.8percent5':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.8, percent=5)

        elif model == 'crop0.7percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.7, percent=1)

        elif model == 'crop0.6percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.6, percent=1)
        elif model == 'crop0.6percent3':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.6, percent=3)
        elif model == 'crop0.6percent5':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.6, percent=5)
        elif model == 'crop0.6percent10':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.6, percent=10)
        elif model == 'crop0.6percent0.1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.6, percent=0.1)

        elif model == 'crop0.5percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.5, percent=1)

        elif model == 'crop0.3percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.3, percent=1)

        elif model == 'crop0.1percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=None, crop_ratio=0.1, percent=1)

        elif model == 'crop0.5delta1.0':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.0, crop_ratio=0.5)
        elif model == 'crop0.9delta1.0':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesMatcher(pb_path, delta=1.0, crop_ratio=0.9)

        elif model == 'patch_cnn_count_crop0.2':
            self.matcher = CnnPatchesMatcher(delta=25, crop_ratio=0.2)
        elif model == 'patch_cnn_count_crop0.3':
            self.matcher = CnnPatchesMatcher(delta=25, crop_ratio=0.3)
        elif model == 'patch_cnn_count_crop0.4':
            self.matcher = CnnPatchesMatcher(delta=25, crop_ratio=0.4)
        elif model == 'patch_cnn_avg_crop0.3':
            self.matcher = CnnPatchesMatcher(delta=None, crop_ratio=0.3)

        elif model == 'softmax_google_global_local_delta0.7_crop.3.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=0.7, crop_ratio_list=[0.3, 0.5, 0.9])
        elif model == 'softmax_google_global_local_delta0.8_crop.3.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=0.8, crop_ratio_list=[0.3, 0.5, 0.9])

        elif model == 'softmax_google_global_local_delta0.9_crop.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=0.9, crop_ratio_list=[0.5, 0.9])
        elif model == 'softmax_google_global_local_delta1.0_crop.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.0, crop_ratio_list=[0.5, 0.9])
        elif model == 'softmax_google_global_local_delta1.1_crop.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.1, crop_ratio_list=[0.5, 0.9])
        elif model == 'softmax_google_global_local_delta1.2_crop.5.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=1.2, crop_ratio_list=[0.5, 0.9])


        elif model == 'softmax_google_global_local_crop0.7_percent1':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=None, crop_ratio_list=0.7, percent=1)
        elif model == 'softmax_google_global_local_crop0.3_percent5':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFLocalGlobalMatcher(pb_path, delta=None, crop_ratio_list=0.3, percent=5)

        elif model == 'casecade_delta0.8':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesCaseCadeMatcher(pb_path, delta=0.8)
        elif model == 'casecade_delta0.85':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesCaseCadeMatcher(pb_path, delta=0.85)
        elif model == 'casecade_delta0.9':
            pb_path = SOFTMAX_GOOGLE_PATH
            self.matcher = TFPatchesCaseCadeMatcher(pb_path, delta=0.9)

        else:
            assert 'Not support Matcher'

        self.k_similar = k_similar
        self.input_op = tf.placeholder(dtype=tf.float64)
        self.top_k_op = tf.nn.top_k(
            self.input_op,
            k=self.k_similar,
            sorted=True,
            name='top_k'
        )
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

    def dist_metrics_with_user_mask(self, dist_metrics, user_info):
        for user, indexes in user_info.items():
            for index in indexes:
                dist_metrics[indexes, index] = 0
        return dist_metrics

    def get_top_k(self, dist_metrics):
        values, indexes = self.sess.run(
            self.top_k_op, feed_dict={self.input_op: dist_metrics})
        return values, indexes


if __name__ == "__main__":
    scene_1_path = 'img/scene_1.jpg'
    scene_2_path = 'img/scene_2.jpg'
    scene_3_path = 'img/scene_3.jpg'
    scene_4_path = 'img/scene_4.jpg'

    scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]
    user_info = {
        'tom': [0, 1],
        'san': [2, 3]
    }

    # create scene recognition serving
    src_serving = SceneRecognitionServing(model='softmax_google_cnn')

    # Get distance metrics
    dist_metrics = src_serving.distance_metrics(scenes, is_norm=True)
    print(dist_metrics)

    dist_metrics = src_serving.dist_metrics_with_user_mask(dist_metrics, user_info)
    print(dist_metrics)

    # Split similar and not similar scene
    similar_indexes = src_serving.filter(dist_metrics, 0.1)

    print(similar_indexes)
