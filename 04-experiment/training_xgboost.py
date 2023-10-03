from xgboost import XGBClassifier
import json
import time
from training_utils import make_data, make_datasets, get_metrics
from training_utils import seeds


def get_model(eval_metric):
    """Get XGBoost model instance
    with the specific evaluation metric for early stopping."""
    return XGBClassifier(
        n_estimators=22_000,
        eval_metric=eval_metric,
        objective='binary:logistic',
        early_stopping_rounds=22,
        random_state=0
    )


def get_predictions(dataset, model):
    """
    Get model predictions.
    """
    return (
        dataset[1],
        model.predict_proba(
            dataset[0], iteration_range=(0, int(model.best_ntree_limit))
        )[:, 1]
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
        training, validation, testing = make_datasets(
            make_data(seed)
        )
        #
        model = get_model(eval_metric)
        #
        start = time.process_time()
        model.fit(
            X=training[0], y=training[1],
            eval_set=[(validation[0], validation[1])],
            verbose=False
        )
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
        open('out/xgboost-{}.json'.format(eval_metric), 'w')
    )


if __name__ == '__main__':
    for eval_metric in ('auc', 'aucpr', 'logloss'):
        job(eval_metric)
