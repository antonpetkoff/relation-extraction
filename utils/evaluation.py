from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import OneHotEncoder


def compute_score(predicted_labels, gold_labels, labels=[], average='weighted'):
    accuracy = accuracy_score(gold_labels, predicted_labels)
    precision = precision_score(gold_labels, predicted_labels, average=average)
    recall = recall_score(gold_labels, predicted_labels, average=average)
    f1 = f1_score(gold_labels, predicted_labels, average=average)
    # average_precision = average_precision_score(
    #     y_true=one_hot_encoded_gold_labels.toarray(),
    #     y_score=one_hot_encoded_predicted_labels.toarray(),
    #     average='micro'
    # )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'average_precision': average_precision,
    }


def compute_precision_recall_curve(y_true, y_pred, classes):
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(classes)

    one_hot_encoded_gold_labels = one_hot_encoder.transform(y_true)
    one_hot_encoded_predicted_labels = one_hot_encoder.transform(y_pred)

    precision, recall, threshold = precision_recall_curve(
        y_true=one_hot_encoded_gold_labels.reshape(-1, 1).toarray(),
        probas_pred=one_hot_encoded_predicted_labels.reshape(-1, 1).toarray(),
    #     pos_label=1
    )
    return precision, recall, threshold
