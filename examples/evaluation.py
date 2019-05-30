import os
import sys

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../'))

from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.utils.fixes import signature
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from scr.utils import ScenceDataset
from scr.serving import SceneRecognitionServing

import time

threshold_list = [i * 0.05 for i in range(20)]
output_log = [[thresh] for thresh in threshold_list]


def draw_pr_curve(precision, recall, color='b', title='PR Curve', marker=">"):
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    # plt.step(recall, precision, color=color, alpha=0.2,
    #          where='post')

    plt.plot(recall, precision, color=color)
    # plt.scatter(recall, precision, color=color, marker=marker)
    # plt.fill_between(recall, precision, alpha=0.2, color=color, **step_kwargs)

    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])


def precision(y_true, y_pred):
    each_class_precision = precision_score(y_true, y_pred, average=None)
    ap = precision_score(y_true, y_pred, average='macro')

    return each_class_precision[1]


def recall(y_true, y_pred):
    each_class_recall = recall_score(y_true, y_pred, average=None)
    ar = recall_score(y_true, y_pred, average='macro')
    return each_class_recall[1]


def eval_model(data_root, model, k_similar=3):
    db = ScenceDataset()
    db.load_scene_db(data_root)

    print(model)
    # create scene recognition serving
    src_serving = SceneRecognitionServing(model=model, k_similar=k_similar)

    preds = dict()
    labels = dict()

    confus_metric = dict()
    for i, (store_name, img_ids) in enumerate(db.image_ids_group_by_store.items()):
        # if store_name != '66_inscene': continue
        print('processing store : {} ({}/{})'.format(store_name, i, len(db.image_ids_group_by_store.keys())))
        img_list = [db.image_paths[img_id] for img_id in img_ids]
        _label = [db.image_labels[img_id] for img_id in img_ids]

        user_info = {'fake_user':[a for a in range(len(_label)) if _label[a]==1]}
        dist_metrics = src_serving.distance_metrics(img_list)
        dist_metrics = src_serving.dist_metrics_with_user_mask(dist_metrics, user_info)

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

            # print('a:{},b:{},c:{},d:{}'.format(cost_a,cost_b,cost_c,cost_d))

    ap_list = []
    ar_list = []

    for i, thresh in enumerate(threshold_list):
        tn = confus_metric[thresh]['tn']
        fp = confus_metric[thresh]['fp']
        fn = confus_metric[thresh]['fn']
        tp = confus_metric[thresh]['tp']

        # tn, fp, fn, tp = confusion_matrix(_label, _pred).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        # ap = precision(_label, _pred)
        ap_list.append(tpr)
        # ar = recall(_label, _pred)
        ar_list.append(fpr)

        output_log[i].extend([str(tpr), str(fpr)])

    return [ap_list, ar_list]


def eval_all():
    data_root = '/root/data/new_restaurant3_dataset_/'
    # model_types = ['cnn', 'softmax_cnn', 'triplet_cnn', 'patch_cnn', 'softmax_patch_cnn', 'triplet_patch_cnn']
    model_types = [
        'softmax_google_global_local_delta0.9_crop.5.9',
        'softmax_google_global_local_delta1.0_crop.5.9',
        'softmax_google_global_local_delta1.1_crop.5.9',
        'softmax_google_global_local_delta1.2_crop.5.9',
        'softmax_google_cnn']

    colors = ['k', 'g', 'r', 'r', 'k', 'k', 'k', 'k']
    markers = ["o", "+", "^", "s", "x", "+", "o", "^"]
    # colors = ['b', 'b', 'b', 'g', 'g', 'g']
    # markers = ["o", "+", "^", "o", "+", "^", ]

    results = {}
    for i, model in enumerate(model_types):
        res = eval_model(data_root, model)
        results[model] = res

    for model, res in results.items():
        i = model_types.index(model)
        ap, ar = res[:]
        draw_pr_curve(ap, ar, color=colors[i], marker=markers[i])
    save_path = 'pr_curve.png'
    plt.savefig(save_path)
    plt.cla()

    import csv
    with open("result.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in output_log:
            writer.writerow(row)


if __name__ == "__main__":
    eval_all()
