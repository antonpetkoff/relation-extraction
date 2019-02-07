import time

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, make_scorer

from models import RandomBaseline, LogRegBaseline, Bert
from utils import aggregate_results, write_results_to_file, get_n_jobs, config_local, humanize_time, flatten
from data_utils import split_data, get_eval_data, get_test_data, persist_predictions

TRAIN_DIR = 'data/train/'
PREDICTIONS_OUTPUT_DIR = 'predictions/'

def main(estimator=LogRegBaseline,
         cv_split=5,
         with_cross_validation=True,
         with_validation=False,
         test_data_dir=None,
         train_validation_ratio=0.8,
         stratify=False):
    x, y = get_eval_data(TRAIN_DIR)
    train_x, validation_x, train_y, validation_y = split_data(x, y, ratio=train_validation_ratio, shuffle=True, stratify_by_labels_ratio=stratify)
    train_x, validation_x, train_y, validation_y = flatten(train_x), flatten(validation_x), flatten(train_y), flatten(validation_y)
    train_validation_x = train_x + validation_x
    train_validation_y = train_y + validation_y

    clf = estimator() if estimator else LogRegBaseline()

    cv_result = cross_validation(clf, train_validation_x, train_validation_y, cv_split) if with_cross_validation else None

    val_result = validation(clf, train_x, train_y, validation_x, validation_y) if with_validation else None

    results = aggregate_results(clf_params=clf.params, cv=cv_result, val=val_result)

    if config_local().get('persist_results', False): write_results_to_file(results)

    if test_data_dir: test(clf, train_validation_x, train_validation_y, test_data_dir, PREDICTIONS_OUTPUT_DIR)

def cross_validation(clf, x, y, cv_split):
    print('Cross-validating model... Samples size:', len(y))

    t_start = time.time()
    cv = cross_validate(estimator=clf, X=x, y=y, fit_params={}, cv=cv_split, scoring=make_scorer(f1_score, average='micro'), n_jobs=get_n_jobs(), return_train_score=True)
    cv['size'] = len(y)
    cv['folds'] = cv_split
    t_end = time.time()

    print('Cross-validating finished... Time taken:', humanize_time(t_end - t_start))

    return cv

def validation(clf, train_x, train_y, validation_x, validation_y):
    print('Evaluating model... Train/Validation samples: %d / %d' % (len(train_y), len(validation_y)))

    t_start = time.time()
    clf.fit(train_x, train_y)
    predictions = clf.predict(validation_x)
    t_end = time.time()

    print('Evaluating finished... Time taken:', humanize_time(t_end - t_start))

    return {
        'f1': f1_score(validation_y, predictions, average='micro'),
        'time': t_end - t_start,
        'train_size': len(train_y),
        'validation_size': len(validation_y),
    }

def test(clf, train_x, train_y, test_data_dir, predictions_output_dir):
    test_x = get_test_data(test_data_dir)
    test_x = flatten(test_x)

    print('Predicting...Train/Test samples: %d / %d' % (len(train_y), len(test_x)))

    t_start = time.time()
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    t_end = time.time()

    persist_predictions(clf, train_x, predictions, predictions_output_dir)

    print('Predicting finished... Time taken:', humanize_time(t_end - t_start))

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Total time: %s" % humanize_time(t_end - t_start))
