import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from scr.utils import ScenceDataset
from scr.serving import SceneRecognitionServing

import time

threshold_list = [i * 0.02 for i in range(50)]
output_log = [[thresh] for thresh in threshold_list]


def draw_pr_curve(precision, recall, color='b', title='ROC Curve', marker=">"):
    # plt.plot(recall, precision, color=color)
    plt.plot(recall, precision, color=color, marker=marker)

    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])


def eval_model(data_root, model, k_similar=3):
    db = ScenceDataset()
    db.load_scene_db(data_root)

    print(model)
    # create scene recognition serving
    src_serving = SceneRecognitionServing(model=model, k_similar=k_similar)

    confus_metric = dict()
    for i, (store_name, img_ids) in enumerate(db.image_ids_group_by_store.items()):
        print('processing store : {} ({}/{})'.format(store_name, i, len(db.image_ids_group_by_store.keys())))
        img_list = [db.image_paths[img_id] for img_id in img_ids]
        _label = [db.image_labels[img_id] for img_id in img_ids]

        user_info = {'fake_user': [a for a in range(len(_label)) if _label[a] == 1]}
        dist_metrics = src_serving.distance_metrics(img_list)
        dist_metrics = src_serving.dist_metrics_with_user_mask(dist_metrics, user_info)

        # For testing different threshold on predict distance metrix
        for thresh in threshold_list:

            # start_a = time.time()
            if thresh not in confus_metric:
                confus_metric[thresh] = {
                    'tn': 0,
                    'fp': 0,
                    'fn': 0,
                    'tp': 0,
                }
            similar_indexes = src_serving.filter(dist_metrics, thresh=thresh)
            # cost_a = time.time()-start_a

            # start_b = time.time()
            _preds = [1 for i in range(len(img_ids))]
            for sim_index in similar_indexes:
                _preds[sim_index] = 0

            # cost_b = time.time()-start_b

            # start_c = time.time()
            tn, fp, fn, tp = confusion_matrix(_label, _preds).ravel()
            # cost_c = time.time() - start_c

            # start_d = time.time()
            confus_metric[thresh]['tn'] += tn
            confus_metric[thresh]['fp'] += fp
            confus_metric[thresh]['fn'] += fn
            confus_metric[thresh]['tp'] += tp
            # cost_d = time.time() - start_d

    tpr_list = []
    fpt_list = []

    for i, thresh in enumerate(threshold_list):
        tn = confus_metric[thresh]['tn']
        fp = confus_metric[thresh]['fp']
        fn = confus_metric[thresh]['fn']
        tp = confus_metric[thresh]['tp']

        # tn, fp, fn, tp = confusion_matrix(_label, _pred).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpt_list.append(fpr)

        output_log[i].extend([str(tpr), str(fpr)])

    return [tpr_list, fpt_list]


def eval_all():
    data_root = '/root/data/new_restaurant3_dataset_/'
    results = {}

    model_types = [
        'pretrained',
        'resnet50',
        'mle',
        'mle_crop.3',
        'mle_crop.5',
        'mle_crop.7',
        'mle_crop.9',
        'mle_crop.3.5',
        'mle_crop.3.5.7',
        'mle_crop.5.7',
        'mle_crop.5.7.9',
        'mle_crop.7.9',
        'mle_delta_0.8',
        'mle_delta_0.9',
        'mle_delta_1.0',
        'mle_delta_1.1',
        'mle_delta_1.2',
        'facenet',
    ]
    colors = ['k', 'k', 'k', 'k', 'k', 'k', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r', 'r']
    markers = ["o", "+", "^", "s", "x", "+", "o", "+", "^", "s", "x", "+", "o", "+", "^", "s", "x", "+"]

    for i, model in enumerate(model_types):
        res = eval_model(data_root, model)
        results[model] = res
    for model, res in results.items():
        i = model_types.index(model)
        ap, ar = res[:]
        draw_pr_curve(ap, ar, color=colors[i], marker=markers[i])

    """
    # NOTE: For experiment on testing top-k matching images performance
    """

    # model_types = [
    #     'pretrained',
    #     'resnet50',
    #     'facenet',
    #     'pyramid_bottleneck_google_cnn',
    #     'triplet_google_cnn',
    # ]

    # colors = ['r', 'g', 'b', 'k', 'g', 'g']
    # markers = ["o", "+", "^", "o", "+", "^", ]

    #
    #
    # k_list = [2,3,5,7]
    # model = 'fg_no_attention_google_cnn'
    # model_types = []
    # for k in k_list:
    #     res = eval_model(data_root, model,k_similar=k)
    #     results[model+'_k{}'.format(k)] = res
    #     model_types.append(model+'_k{}'.format(k))
    #
    #
    # for model, res in results.items():
    #     i = model_types.index(model)
    #     ap, ar = res[:]
    #     draw_pr_curve(ap, ar, color=colors[i], marker=markers[i])
    # plt.legend(['v=1', 'v=2', 'v=4', 'v=6'])

    save_path = 'exp0.png'
    plt.savefig(save_path)
    plt.cla()

    import csv
    """
    All TPR and FPR will be saved in result.csv ,
    Can use script examples/experiments_ploter.py to choice drawing curve combinations.
    """
    with open("result.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in output_log:
            writer.writerow(row)


if __name__ == "__main__":
    eval_all()
