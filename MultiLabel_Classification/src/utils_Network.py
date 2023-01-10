import pandas as pd
import numpy as np
from utils import *
import random
import sys
from sklearn.utils.class_weight import compute_class_weight


def parse_input(filename_mat,labs):

    count_mat_geno_sub = pd.read_csv(filename_mat, sep="\t", index_col=0)

    if set(labs.index.tolist())==set(count_mat_geno_sub.index.tolist()):
        count_mat_geno_sub = count_mat_geno_sub.loc[labs.index.tolist(),:]
        return count_mat_geno_sub

    else:
        sys.exit("labels and input are different")


def separate_testset(labs, mat):

    test_labs = []

    for mut in labs.columns.tolist():

        random.seed(7)
        temp = random.sample(labs.index[labs[mut] == 1].tolist(),round(len(labs.index[labs[mut] == 1].tolist()) * 0.15))
        test_labs = test_labs + temp

    test_labs = list(set(test_labs))
    random.seed(7)
    test_labs_sh = random.sample(test_labs,len(test_labs))

    testY = labs.loc[test_labs_sh, ]
    testX = mat.loc[test_labs_sh, ]
    touseY = labs.loc[~labs.index.isin(test_labs_sh)]
    touseX = mat.loc[~mat.index.isin(test_labs_sh)]

    return touseX, testX, touseY, testY


def cells_to_probs_class(net, cells, mode, thresh=0.5):

    output = net(cells.float())

    if mode=="train":
        y_hat_class = []
    else:
        y_hat_class = np.where(output.cpu().detach().numpy()<thresh, 0, 1)

    return y_hat_class, output


def compute_weight_class(labs,unique_values=[0.,1.]):

    return list(compute_class_weight('balanced', unique_values,labs))





