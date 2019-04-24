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

from scr.utils import ScenceDataset
from scr.serving import SceneRecognitionServing


def draw_pr_curve(precision, recall, color='b', title='PR Curve'):
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    # plt.step(recall, precision, color=color, alpha=0.2,
    #          where='post')

    # plt.plot(recall, precision, color=color)
    plt.scatter(recall, precision, color=color)
    # plt.fill_between(recall, precision, alpha=0.2, color=color, **step_kwargs)

    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])


def draw(precision, recall, precision_class_0, recall_class_0,
         precision_class_1, recall_class_1):
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))

    lines = []
    labels = []

    l, = plt.plot(precision, recall, color='darkorange', lw=2)
    lines.append(l)
    labels.append('Precision-recall')

    l, = plt.plot(precision_class_0, recall_class_0, color='navy', lw=2)
    lines.append(l)
    labels.append('Precision-recall Class 0 ')

    l, = plt.plot(precision_class_1, recall_class_1, color='turquoise', lw=2)
    lines.append(l)
    labels.append('Precision-recall Class 1 ')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig('p-r.png')


def precision(y_true, y_pred):
    each_class_precision = precision_score(y_true, y_pred, average=None)
    ap = precision_score(y_true, y_pred, average='macro')
    return ap, each_class_precision


def recall(y_true, y_pred):
    each_class_recall = recall_score(y_true, y_pred, average=None)
    ar = recall_score(y_true, y_pred, average='macro')
    return ar, each_class_recall


def eval_model(data_root, model):
    db = ScenceDataset()
    db.load_scene_db(data_root)

    # create scene recognition serving
    src_serving = SceneRecognitionServing(model=model)

    threshold_list = [i * 0.05 for i in range(20)]

    preds = dict()
    labels = dict()
    for i, (store_name, img_ids) in enumerate(db.image_ids_group_by_store.items()):
        print('processing store : {} ({}/{})'.format(store_name, i, len(db.image_ids_group_by_store.keys())))
        img_list = [db.image_paths[img_id] for img_id in img_ids]

        dist_metrics = src_serving.distance_metrics(img_list)

        _ap = []
        _ar = []
        for thresh in threshold_list:
            if thresh not in preds:
                preds[thresh] = []
                labels[thresh] = []
            similar_indexes = src_serving.filter(dist_metrics, thresh=thresh)

            _preds = [1 for i in range(len(img_ids))]
            for sim_index in similar_indexes:
                _preds[sim_index] = 0
            _label = [db.image_labels[img_id] for img_id in img_ids]

            preds[thresh].extend(_preds)
            labels[thresh].extend(_label)
        #     print('thresh:{}'.format(thresh))
        #     # print('pred:{}'.format(_preds))
        #     # print('label:{}'.format(_label))
        #
        #
        #     ap = precision(_label, _preds)[1][0]
        #     ar = recall(_label, _preds)[1][0]
        #     print('ap:{}'.format(ap))
        #     print('ar:{}'.format(ar))
        #     _ap.append(ap)
        #     _ar.append(ar)
        # draw_pr_curve(_ap, _ar, color='b')
        # save_path = 'pr_curve_store_{}.png'.format(store_name)
        # print(save_path)
        # plt.savefig(save_path)
        # plt.cla()

    ap_list = []
    ap_list_class_0 = []
    ap_list_class_1 = []
    ar_list = []
    ar_list_class_0 = []
    ar_list_class_1 = []
    for thresh in threshold_list:
        _pred = preds[thresh]
        _label = labels[thresh]

        # _label = _label[:10]
        # _pred = _pred[:10]
        ap, ap_each_clss = precision(_label, _pred)
        ap_list.append(ap)
        ap_list_class_0.append(ap_each_clss[0])
        ap_list_class_1.append(ap_each_clss[1])
        ar, ar_each_clss = recall(_label, _pred)
        ar_list.append(ap)
        ar_list_class_0.append(ar_each_clss[0])
        ar_list_class_1.append(ar_each_clss[1])

    # draw_pr_curve(ap_list, ar_list, save_path='pr_curve.png')
    # draw_pr_curve(ap_list_class_0, ar_list_class_0, save_path='class 0 pr_curve.png')
    # draw_pr_curve(ap_list_class_1, ar_list_class_1, save_path='class 1 pr_curve.png')

    return [ap_list, ar_list, ap_list_class_0, ap_list_class_1, ar_list_class_0, ar_list_class_1]


def eval_all():
    data_root = '/root/data/new_restaurant3/'
    model_types = ['cnn', 'softmax_cnn', 'triplet_cnn', 'patch_cnn', 'softmax_patch_cnn', 'triplet_patch_cnn']
    # model_types = ['cnn']
    colors = ['b', 'g', 'r', 'y', 'k', 'c']

    results = {}
    for i, model in enumerate(model_types):
        res = eval_model(data_root, model)
        results[model] = res

    for model, res in results.items():
        ap, ar, ap_0, ap_1, ar_0, ar_1 = res[:]
        draw_pr_curve(ap, ar, color=colors[i])
    save_path = 'pr_curve.png'
    plt.savefig(save_path)
    plt.cla()

    classes = ['all', 'class_0', 'class_1']
    for i in range(3):
        for model_no, (model, res) in enumerate(results.items()):
            ap, ar = res[i * 2:i * 2 + 2]
            draw_pr_curve(ap, ar, color=colors[model_no])
        save_path = 'pr_curve_{}.png'.format(classes[i])
        plt.savefig(save_path)
        plt.cla()


if __name__ == "__main__":
    # data_root = '/root/data/new_restaurant3/'
    # eval_model(data_root, 'cnn')
    # quit()

    eval_all()
