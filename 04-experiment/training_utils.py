
import numpy as np
import random
np.random.seed(0)
random.seed(0)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


seeds = [978, 672, 821, 445, 488, 449, 753, 962, 874, 287, 257, 598, 100, 136, 305, 376, 548, 229, 265, 425]
# seeds from previous experiments


def make_data(random_state):
    """
    Create synthetic dataset according to #4 experiment.
    """
    return make_classification(
        n_samples=50_000_000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        weights=(0.9995, 0.0005), class_sep=0.1, flip_y=0.0001,
        random_state=random_state)


def make_datasets(data):
    """
    Split synthetic dataset into training, validation and testing parts
    in the 50/25/25% proportion.
    """
    x_training, x, y_training, y = train_test_split(
        data[0],
        data[1],
        train_size=0.5,
        shuffle=True,
        stratify=data[1],
        random_state=0
    )
    #
    x_validation, x_testing, y_validation, y_testing = train_test_split(
        x,
        y,
        train_size=0.5,
        shuffle=True,
        stratify=y,
        random_state=0
    )
    #
    return (
        (x_training, y_training),
        (x_validation, y_validation),
        (x_testing, y_testing)
    )


def get_metrics(target, scores):
    """
    Get the estimates of AUC ROC and AUC PR.
    """
    return (
        roc_auc_score(target, scores),
        average_precision_score(target, scores),
    )
