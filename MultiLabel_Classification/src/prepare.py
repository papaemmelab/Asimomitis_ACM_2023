from utils import *
from utils_Network import *
import random
import sys
import pandas as pd


def create_datasets(config):

    print("\nPrepare the training and testing datasets\n")

    ngenes = config["ngenes"]
    print("Number of genes used in the model: %d" % ngenes)

    mydata_dir = config["data_dir"]
    print("My Results Data Directory: %s" % mydata_dir)

    scripts_dir = config["scripts_dir"]
    print("My scripts Directory: %s" % scripts_dir)

    patient = config["patient"]
    print("Patient: %s" % patient)

    mutcols = config["mutcols"]

    device = config["device"]
    print("Running on: %s" % device)

    local_hierarchies = {0: ["IDH2_R140", "IDH2_R172", "IDH1_R132", "DNMT3A_R882"],
                         1: ["NPM1_W288", "SRSF2_P95"],
                         2: ["NRAS_G12", "KRAS_G12"],
                         3: ["has_dupli_chr14", "has_dupli_chr8", "has_dupli_chr10", "has_dupli_chr6", "has_dupli_chr1"],
                         4: ["WT"]
                        }

    for lh in list(local_hierarchies.keys()):
        local_hierarchies[lh] = [lhi for lhi in local_hierarchies[lh] if lhi in mutcols]

    filename_output = scripts_dir + "patient_spec_labels.csv"
    labels = pd.read_csv(filename_output, sep="\t", index_col=0).astype(int).loc[:,mutcols]

    filename_input_mat = scripts_dir + "+patient_spec_gene_expr.csv"
    count_mat_geno_sub = parse_input(filename_input_mat, labels)

    if (labels.index == count_mat_geno_sub.index).all():
        print("Labels are correctly assigned and datasets are prepared")
        touseX, testX, touseY, testY = separate_testset(labels, count_mat_geno_sub)

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

    if [item for sublist in list(local_hierarchies.values()) for item in sublist] != touseY.columns.tolist():
        sys.exit("Inconsistency between the order of the mutations in the labels and the hierarchies")

    print("here4")
    return_dict = {
        'touseX': touseX,
        'touseY': touseY,
        'testX': testX,
        'testY': testY,
        'classes': touseY.columns.tolist(),
        'local_hierarchies': local_hierarchies
    }

    return return_dict
