from fit import *
import os
import argparse
import sys
import torch
from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import pandas as pd
import json
import numpy as np


def main():

    print("\nWelcome to scBinary!")

    init()

    metric_choices_min = ['loss']
    metric_choices_max = ['accuracy','precision','recall','TNR','NPV','f1']

    parser = argparse.ArgumentParser(description='Specify the DNN model.')
    parser.add_argument('--scaling', choices=['MaxAbs','Normalization01','Normalization-11','Standardization','no'], help='Scaling applied', default="no")
    parser.add_argument('--metric', choices=metric_choices_min+metric_choices_max, help='Metric used for optimization', default="loss")
    parser.add_argument('--hlayers', type=int, nargs='+', help ='Size of hidden layers')
    parser.add_argument('--dropout_layers',type=int, nargs='+', help='Positions of Dropout layers within Hidden Layers',default=0)
    parser.add_argument('--epochs', type=int, help='Number of epochs used in the training of the DNN', default=100)
    parser.add_argument('--early_stopping',choices=['yes', 'no'], help='Apply early stopping during fitting', default='yes')
    parser.add_argument('--patience', type=int, help='Number of epochs with no improvement in early stopping', default=0)
    parser.add_argument('--num_samples', type=int, help='Number of samplings from hyperparameter space', default=64)
    parser.add_argument('--grace_period', type=int, help='Grace Period', default=100)
    parser.add_argument('--cpus', type=int, help='Number of cpus to allocate per trial', default=4)
    parser.add_argument('--gpus', type=float, help='Number of gpus to allocate per trial', default=0)
    parser.add_argument('--ngenes', type=int, help='Number of used in the model', default=1000)
    parser.add_argument('--perc_split', type=float, help='perc_split', default=0.06)

    print("CUDA VISIBLE DEVICES")
    print(os.environ.get('CUDA_VISIBLE_DEVICES'))

    args = parser.parse_args()

    scaling = args.scaling
    hlayers = list(args.hlayers)
    epochs = args.epochs
    early_st = args.early_stopping
    patience = args.patience
    ngenes = args.ngenes
    mymetric = args.metric
    num_samples = args.num_samples
    grace_period = args.grace_period
    cpus = args.cpus
    gpus = args.gpus
    perc_split = args.perc_split

    if mymetric in metric_choices_min:

        mymode="min"
    else:
        mymode="max"

    hlstr = ','.join(str(e) for e in hlayers)

    if isinstance(args.dropout_layers,int):
        dropoutlstr = str(args.dropout_layers)
        dropout_layers = [args.dropout_layers]
    else:
        dropout_layers = list(args.dropout_layers)
        dropoutlstr = ','.join(str(e) for e in dropout_layers)

    dropout_layers.sort()

    if early_st=='yes':
        if patience==0:
            patience=7
    else:
        if patience!=0:
            sys.exit('\nUse of early stopping is not selected but patience is specified. Please review the arguments. Exiting...\n')

    use_gpu = False #torch.cuda.is_available()

    if use_gpu == True:
        print("\nUsing GPU")
        device = 'cuda'
    else:
        print("\nUsing CPU")
        device = 'cpu'

    scripts_dir = os.getcwd()+"/"

    exper_name = 'Run_Bin_'+str(perc_split)+ mymetric + '_scaling'+scaling+ "_Layers" + hlstr + '_dropoutlayers'+ dropoutlstr +\
                 "_epochs" + str(epochs) + '_pat' + str(patience) + '_ngenes' + str(ngenes) + \
                 '_cpus' + str(cpus) + '_gpus'+ str(gpus)+ '_numsamples'+str(num_samples)+ \
                 '_graceperiod' + str(grace_period)+ '_compHyb_dev' + device

    data_dir = scripts_dir+"../results/NN/model/"+exper_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    config = {
        "batch_size": tune.choice([8, 16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "scaling": scaling,
        "weight_decay": tune.choice([0, 0.01, 0.001, 0.0001]),
        "weighted_loss": tune.choice(['yes','no']),
        "dropout_rate": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
        "optimization": tune.choice(['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'ASGD', 'RMSprop', 'Rprop', 'SGD']),
        "hlayers": hlayers,
        "epochs": epochs,
        "activation":'relu',
        "early_st": early_st,
        "patience": patience,
        "ngenes": ngenes,
        "dropout_layers": dropout_layers,
        "data_dir": data_dir,
        "scripts_dir": scripts_dir,
        "grace_period": grace_period,
        "perc_split": perc_split,
        "device": device
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=mymetric,
        mode=mymode,
        max_t=epochs,
        grace_period=grace_period,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "precision","recall","NPV","TNR","f1","training_iteration"])

    result = tune.run(
        partial(fit,data_dir=data_dir),
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=scripts_dir+"../results/NN/ray_results",
        name=exper_name,
        global_checkpoint_period=np.inf,
        checkpoint_at_end=True)

    best_config = result.get_best_config(mymetric, mymode, "last-10-avg")
    bestlog_path = result.get_best_logdir(mymetric, mymode, "last-10-avg")
    progress_df = pd.read_csv(bestlog_path + "/progress.csv")
    epochs_totrain = progress_df.shape[0]

    outfile=data_dir+'/best_configuration_parameters.json'

    with open(outfile, 'w') as fp:
        json.dump(best_config, fp,sort_keys=True, indent=4)

    print('\nThe parameters of the best model are saved in the %s' % outfile)
    print("\nFitting the best DNN model on whole training dataset...")

    trained_model, datadict, writer, scaler_all, device, mydata_path, resultsfolder_path = train_all(best_config,epochs_totrain)
    print("\nModel was trained.")

    final_path = mydata_path+'/'+resultsfolder_path
    print("See training, validation & visualizations here:'%s'\n" % final_path)


if __name__ == "__main__":
    main()









