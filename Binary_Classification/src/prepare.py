from utils_Network import *
import random
import pandas as pd


def create_datasets(config):

    print("\nPrepare the training and testing datasets\n")

    ngenes = config["ngenes"]
    print("Number of genes used in the model: %d" % ngenes)

    mydata_dir = config["data_dir"]
    print("My Results Data Directory: %s" % mydata_dir)

    scripts_dir = config["scripts_dir"]
    print("My scripts Directory: %s" % scripts_dir)

    perc_split = config["perc_split"]
    print("Test split percentage: %s" % perc_split)

    device = config["device"]
    print("Running on: %s" % device)

    filename_output = scripts_dir + "labels.csv"
    labels = pd.read_csv(filename_output, sep="\t", index_col=0).astype(int)[['WT']]
    labels = labels + 1
    labels = labels.replace(2, 0)
    labels.columns = ['Mal']

    filename_input_mat = scripts_dir + "gene_expr.csv"
    count_mat_geno_sub = parse_input(filename_input_mat, labels)

    if (labels.index == count_mat_geno_sub.index).all():
        print("Labels are correctly assigned and datasets are prepared")
        touseX, testX, touseY, testY = separate_testset(labels, count_mat_geno_sub,perc_split)

    print(testX.shape)

    if (touseX.index == touseY.index).all():
        print("Training data is correctly assigned")

        indx = touseY.index.tolist()
        random.seed(7)
        indx_sh = random.sample(indx, len(indx))
        touseX = touseX.loc[indx_sh,]
        touseY = touseY.loc[indx_sh,]

        if (touseX.index == touseY.index).all():
            print("Training data is correctly assigned after shuffling")

    if (testX.index == testY.index).all():
        print("Testing data is correctly assigned\n")

    print("here4")

    return_dict = {
        'touseX': touseX,
        'touseY': touseY,
        'testX': testX,
        'testY': testY,
        'classes': touseY.columns.tolist(),
    }

    return return_dict
