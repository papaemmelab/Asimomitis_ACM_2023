import numpy as np


def tnr_fn(fp, tn):
    if (tn + fp) != 0:
        return tn / (tn + fp)
    else:
        return np.nan


def npv_fn(tn, fn):
    if (tn + fn) != 0:
        return tn / (tn + fn)
    else:
        return np.nan


