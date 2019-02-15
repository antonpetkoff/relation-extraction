import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import visualization


def calc_prec_recall_f1(y_actual, y_pred, none_id='NA', eps=0.00000001):
    pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0

    for i in range(len(y_actual)):
        if y_actual[i] != none_id:
            pos_gt += 1.0

    for i in range(len(y_pred)):
        if y_pred[i] != none_id:
            pos_pred += 1.0                    # classified as pos example (Is-A-Relation)
            if y_pred[i] == y_actual[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + eps)
    recall    = true_pos / (pos_gt + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def calc_average_precision_area(y_actual, y_pred_probs, classes):
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(classes.reshape(-1, 1))

    y_hot = np.array(one_hot_encoder.transform(y_actual).todense())
    logit_list = y_pred_probs

    # e[:-1] removes the last class which is NA
    y_true   = np.array([e[:-1] for e in y_hot]).reshape((-1))
    y_scores = np.array([e[:-1] for e in logit_list]).reshape((-1))
    area_pr  = average_precision_score(y_true, y_scores)
    return area_pr, y_true, y_scores


def plot_precision_recall_curve(y_true, y_scores, name):
    precision,recall,threshold = precision_recall_curve(y_true, y_scores)
    baselines_path             = '../baselines_pr/'

    this_model_dir = baselines_path + name

    if not os.path.exists(this_model_dir):
        os.makedirs(this_model_dir)

    saved_precision_filename = this_model_dir + '/precision.npy'
    with open(saved_precision_filename, 'wb') as f:
        np.save(saved_precision_filename, precision)
    print('Precision points array saved at: {}'.format(saved_precision_filename))

    saved_recall_filename = this_model_dir + '/recall.npy'
    with open(saved_recall_filename, 'wb') as f:
        np.save(saved_recall_filename, recall)
    print('Recall points array saved at: {}'.format(saved_recall_filename))

    plt.plot(recall[:], precision[:], label=name, color ='red', lw=1, marker = 'o', markevery = 0.1, ms = 6)

    base_list = [# 'LogRegAll_CW','LogRegAll', 'CNN',
                 'RESIDE', 'BGWA', 'PCNN+ATT', 'PCNN', 'MIMLRE', 'MultiR', 'Mintz']
    color     = [# 'magenta', 'orange', 'blue',
                 'black', 'purple', 'darkorange', 'green', 'xkcd:azure', 'orchid', 'cornflowerblue']
    marker    = [# 'p', 'h', '8',
                 '+', 'd', 's', '^', '*', 'v', 'x', 'h']
#     plt.ylim([0.3, 1.0])
#     plt.xlim([0.0, 0.45])

    for i, baseline in enumerate(base_list):
        precision = np.load(baselines_path + baseline + '/precision.npy')
        recall    = np.load(baselines_path + baseline + '/recall.npy')
        plt.plot(recall, precision, color = color[i], label = baseline, lw=1, marker = marker[i], markevery = 0.1, ms = 6)

    plt.xlabel('Recall',    fontsize = 14)
    plt.ylabel('Precision', fontsize = 14)
    plt.legend(loc="upper right", prop = {'size' : 12})
    plt.grid(True)
    plt.tight_layout()

    plot_path = '../plots/pr_curve_{}.png'.format(name)
    plt.savefig(plot_path)
    print('Precision-Recall plot saved at: {}'.format(plot_path))
    return plt


def get_pscore(y_true, y_scores, label='P@N Evaluation'):
    allprob = np.reshape(np.array(y_scores), (-1))
    allans  = np.reshape(y_true, (-1))
    order   = np.argsort(-allprob)

    def p_score(n):
        corr_num = 0.0
        for i in order[:n]:
            corr_num += 1.0 if (allans[i] == 1) else 0
        return corr_num / n

    return p_score(100), p_score(200), p_score(300)


def evaluate(y_actual, y_pred, y_pred_probs, classes, name):
    precision, recall, f1 = calc_prec_recall_f1(y_actual, y_pred)
    ap_area, y_true, y_scores = calc_average_precision_area(y_actual, y_pred_probs, classes)

    plt.rcParams["figure.figsize"] = (12, 10)
    plot_precision_recall_curve(y_true, y_scores, name).show()

    cnf_matrix = confusion_matrix(
        y_true=y_actual,
        y_pred=y_pred,
        labels=classes
    )

    plt.rcParams["figure.figsize"] = (28, 28)
    visualization.plot_confusion_matrix(
        cm=cnf_matrix,
        classes=classes,
        name=name,
        normalize=True,
        title='Normalized confusion matrix'
    ).show()

    accuracy = accuracy_score(y_actual, y_pred)

    p100, p200, p300 = get_pscore(y_true, y_scores)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap_area': ap_area,
        'p@100': p100,
        'p@200': p200,
        'p@300': p300
    }

    results_path = '../results/{}.json'.format(name)
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print('Saved results JSON at: {}'.format(results_path))

    return results
