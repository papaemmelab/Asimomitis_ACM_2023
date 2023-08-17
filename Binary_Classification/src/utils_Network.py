import pandas as pd
import numpy as np
import random
from collections import Counter
import torch
import sys


def parse_input(filename_mat, labs):

    count_mat_geno_sub = pd.read_csv(filename_mat, sep="\t", index_col=0)

    if set(labs.index.tolist())==set(count_mat_geno_sub.index.tolist()):
        count_mat_geno_sub = count_mat_geno_sub.loc[labs.index.tolist(),:]
        return count_mat_geno_sub

    else:
        sys.exit("labels and input are different")


def separate_testset(labs, mat, perc_split=0.06):

    random.seed(7)
    temp1 = random.sample(labs.index[labs['Mal'] == 0].tolist(), round(len(labs.index[labs['Mal'] == 0].tolist())*perc_split)) 

    random.seed(7)
    temp2 = random.sample(labs.index[labs['Mal'] == 1].tolist(), round(len(labs.index[labs['Mal'] == 1].tolist())*perc_split)) 
    test_labs = temp1 + temp2

    random.seed(7)
    test_labs_sh = random.sample(test_labs,len(test_labs))

    testY = labs.loc[test_labs_sh, ]
    testX = mat.loc[test_labs_sh, ]
    touseY = labs.loc[~labs.index.isin(test_labs_sh)]
    touseX = mat.loc[~mat.index.isin(test_labs_sh)]

    return touseX, testX, touseY, testY


def cells_to_probs_class(net, cells, mode):

    if isinstance(cells,np.ndarray):
        output = net(torch.tensor(cells).float())
    else:
        output = net(cells.float())

    if mode=="train":
        y_hat_class = []
    else:
        y_hat_class = np.where(output.cpu().detach().numpy()<0.5, 0, 1)

    return y_hat_class, output


def compute_weight_class(labs):

    table = Counter(labs.reshape(1, -1).tolist()[0])
    w = [table[1] / table[0],1]

    return w
