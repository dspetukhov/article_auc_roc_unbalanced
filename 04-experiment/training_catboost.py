from catboost import CatBoostClassifier, Pool
import json
import time
from training_utils import make_data, make_datasets, get_metrics
from training_utils import seeds


def get_pools(seed):
    """
    Create CatBoost pools from a dataset
    divided into training, validation and testing (holdout) parts.
    """
    training, validation, testing = make_datasets(
        make_data(seed)
    )
    #
    training = Pool(training[0], label=training[1])
    validation = Pool(validation[0], label=validation[1])
    testing = Pool(testing[0], label=testing[1])
    #
    return training, validation, testing


def get_model(eval_metric):
    """Get CatBoost model instance
    with the specific evaluation metric for early stopping."""
    return CatBoostClassifier(
        iterations=22_000,
        eval_metric=eval_metric,
        objective='Logloss',
        early_stopping_rounds=22,
        random_seed=0,
        use_best_model=True
    )


def get_predictions(dataset, model):
    """
    Get model predictions.
    """
    return (
        dataset.get_label(),
        model.predict_proba(dataset)[:, 1]
    )


def job(eval_metric):
    """Routine for model training and metrics estimation
    on the testing part of the dataset at various seeds."""
    out_metrics = {
        'AUC ROC': [],
        'AUC PR': [],
        'Time': []
    }

    for seed in seeds:
        #
        print(seed)
        #
        training, validation, testing = get_pools(seed)
        #
        model = get_model(eval_metric)
        #
        start = time.process_time()
        model.fit(training, eval_set=validation, verbose=False)
        duration = (time.process_time() - start) / 60
        #
        metrics = get_metrics(
            get_predictions(testing, model)
        )
        #
        out_metrics['AUC ROC'].append(metrics[0])
        out_metrics['AUC PR'].append(metrics[1])
        out_metrics['Time'].append(duration)

    json.dump(
        out_metrics,
        open('out/catboost-{}.json'.format(eval_metric), 'w')
    )


if __name__ == '__main__':
    for eval_metric in ('AUC', 'PRAUC', 'Logloss'):
        job(eval_metric)
