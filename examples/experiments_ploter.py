import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_pr_curve(precision, recall, color='b', title='ROC Curve', marker=">", label='put label'):
    plt.plot(recall, precision, color=color, label=label)
    # plt.plot(recall, precision, color=color, marker=marker, label=label)

    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.7, 1.0])
    plt.xlim([0.05, 0.4])
    # plt.ylim([0.0, 1.0])
    # plt.xlim([0.0, 1.0])


def draw_exp(exp_models, exp_name):
    colors = ['k', 'b', 'g', 'r', 'c', 'm']
    markers = ["o", "+", "^", "s", "x", "+"]
    for i, model in enumerate(exp_models):
        tpr = result_dict[model]['tpr']
        fpr = result_dict[model]['fpr']
        draw_pr_curve(tpr, fpr, color=colors[i], marker=markers[i], label=model)
    plt.legend(exp_models)
    save_path = exp_name + '.png'
    plt.savefig(save_path)
    plt.cla()


result_path = '/root/dennis_code_base/scene_recognition/summary_result.csv'
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
    'mle+msd(.7.9)',
    'mle_delta_0.8',
    'mle_delta_0.9',
    'mle_delta_1.0',
    'mle_delta_1.1',
    'mle_delta_1.2',
    'facenet',
]
#
# result_path = '/root/dennis_code_base/scene_recognition/result.csv'
#
# model_types = model_types = [
#         'pretrained',
#         'softmax_loss',
#         'center_loss(facenet)',
#         'softmax_l2_loss',
#         'triplet_loss',
#     ]

result_dict = {}

for model in model_types:
    result_dict.update({
        model: {
            "tpr": [],
            "fpr": []
        }
    })

with open(result_path, newline='') as csvfile:
    rows = csv.reader(csvfile)

    for row in rows:
        thresh = float(row[0])
        for model_idx in range(len(model_types)):
            model = model_types[model_idx]
            tpr, fpr = row[1 + (model_idx * 2):3 + (model_idx * 2)]
            tpr, fpr = float(tpr), float(fpr)
            result_dict[model]['tpr'].append(tpr)
            result_dict[model]['fpr'].append(fpr)

# exp0_models = ['softmax_loss',
#                'center_loss(facenet)',
#                'softmax_l2_loss',
#                'triplet_loss']
# draw_exp(exp0_models, 'exp0')
# quit()

exp1_models = ['pretrained', 'resnet50']
exp2_models = ['resnet50', 'mle', 'mle+msd(.7.9)']

exp3_models = ['mle',
               'mle_crop.3',
               'mle_crop.5',
               'mle_crop.7',
               'mle_crop.9',
               ]

exp4_models = ['mle',
               'mle_crop.3.5',
               'mle_crop.3.5.7',
               # 'mle_crop.5.7',
               # 'mle_crop.5.7.9',
               'mle+msd(.7.9)',
               ]

exp5_models = ['mle',
               'mle_delta_0.8',
               'mle_delta_0.9',
               'mle_delta_1.0',
               'mle_delta_1.1',
               'mle_delta_1.2',
               ]

exp7_models = [
    'mle',
    'mle+msd(.7.9)',
    'facenet'
]

draw_exp(exp1_models, 'exp1')
draw_exp(exp2_models, 'exp2')
draw_exp(exp3_models, 'exp3')
draw_exp(exp4_models, 'exp4')
draw_exp(exp5_models, 'exp5')
draw_exp(exp7_models, 'exp7')

base_models = [
    'pretrained',
]

draw_exp(base_models, 'base')
