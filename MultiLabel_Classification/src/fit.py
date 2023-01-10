from Network_hierarchy import *
import shutil
from es import *
from prepare import *
import numpy as np
from utils_Network import *
from utils import *
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import metrics
from torch import nn
import torch
import json
from tensorboardX import SummaryWriter
import os
from ray import tune
import sys
from pickle import dump
import pandas as pd


def fit(config, checkpoint_dir=None,data_dir=None):

    datadict = create_datasets(config)

    print("\nThe following arguments were specified\n")

    touseX = datadict['touseX']
    touseY = datadict['touseY']
    classes = datadict['classes']

    hidden_layers_size = config["hlayers"]
    print("Hidden Layers: %s" % str(hidden_layers_size))

    num_epochs = config["epochs"]
    print("Epochs: %d" % num_epochs)

    early_st = config["early_st"]
    print("Early Stopping applied: %s" % early_st)

    patience = config["patience"]
    print("Patience if early stopping is applied: %s" % patience)

    ngenes = config["ngenes"]
    print("Number of genes used in the model: %d" % ngenes)

    learning_rate = config["lr"]
    print("Learning rate: %f" % learning_rate)

    batch_size = config["batch_size"]
    print("Batch Size: %d" % batch_size)

    optimization = config["optimization"]
    print("Optimizer: %s" % optimization)

    scaling = config["scaling"]
    print("Input Scaling: %s" % scaling)

    weighted_loss = config["weighted_loss"]
    print("Weighted Loss applied: %s" % weighted_loss)

    weight_decay = config["weight_decay"]
    print("Weight decay for Optimizer: %f" % weight_decay)

    dropout_layers = config["dropout_layers"]
    print("Dropout layers: %s" % str(dropout_layers))

    dropout_rate = config["dropout_rate"]
    print("Dropout rate: %f" % dropout_rate)

    activation = config["activation"]
    print("Activation function: %s" % activation)

    mydata_dir = config["data_dir"]
    print("My Results Data Directory: %s" % mydata_dir)

    scripts_dir = config["scripts_dir"]
    print("My scripts Directory: %s" % scripts_dir)

    rs = config["rs"]
    print("Random state: %s" % str(rs))

    device = config["device"]
    print("Running on: %s" % device)

    print("\nFitting DNN model...")

    if dropout_layers[-1]>len(hidden_layers_size)-1:
        sys.exit("Dropout layers are more than Hidden Layers. Exiting...")

    num_hidden_layers = len(hidden_layers_size)

    resultsfolder = 'NN_lr'+str(learning_rate)+'_batchsize'+str(batch_size)+'_optimizer'+optimization+'_scaling'+scaling+\
                    '_weightedloss'+weighted_loss+'_weightdecay'+str(weight_decay)+'_activation'+activation+'_dropoutrate'+str(dropout_rate)

    if os.path.isdir(mydata_dir+'/'+resultsfolder):
        dir_path = mydata_dir+'/'+resultsfolder
        shutil.rmtree(dir_path)

    command='tensorboard --logdir='+mydata_dir+'/'+resultsfolder
    print("\nRun this command '"+command+"' and track the progress of fitting interactively in real time!\n")

    writer = SummaryWriter(mydata_dir+'/'+resultsfolder)

    input_size = len(touseX.columns.tolist())
    output_size = len(classes)

    print(classes)
    weight = {}
    for pos, mut in enumerate(classes):
        if -1. in list(np.unique(touseY.values[:, pos])):
            if (0. in list(np.unique(touseY.values[:, pos]))) and (1. in list(np.unique(touseY.values[:, pos]))):
                weight[mut] = compute_weight_class(touseY.loc[touseY[mut]!=-1,:].values[:, pos], unique_values=[0, 1])
                weight[mut].append(0)
            elif 0. in list(np.unique(touseY.values[:, pos])):
                weight[mut] = [1.0, 0.0, 0.0]
            elif 1. in list(np.unique(touseY.values[:, pos])):
                weight[mut] = [0.0, 1.0, 0.0]
        else:
            weight[mut] = compute_weight_class(touseY.values[:, pos], unique_values=[0, 1])

    X_train_pd, Y_train_pd, X_test_pd, Y_test_pd = iterative_train_test_split(touseX.values, touseY.values, test_size=0.2, rs=rs, shf=False)

    X_test_pd = pd.merge(touseX.reset_index(), pd.DataFrame(X_test_pd, columns=touseX.columns.tolist()))
    X_test_pd.set_index('index', inplace=True)

    X_train_pd = pd.merge(touseX.reset_index(), pd.DataFrame(X_train_pd, columns=touseX.columns.tolist()))
    X_train_pd.set_index('index', inplace=True)

    Y_train_pd = pd.DataFrame(Y_train_pd, index=X_train_pd.index.tolist(), columns=classes)

    Y_test_pd = pd.DataFrame(Y_test_pd, index=X_test_pd.index.tolist(), columns=classes)

    X_train_pd.to_csv(mydata_dir+'/'+resultsfolder+'/X_train_pd.csv',sep='\t')
    Y_train_pd.to_csv(mydata_dir+'/'+resultsfolder+'/Y_train_pd.csv',sep='\t')

    X_test_pd.to_csv(mydata_dir+'/'+resultsfolder+'/X_test_pd.csv',sep='\t')
    Y_test_pd.to_csv(mydata_dir+'/'+resultsfolder+'/Y_test_pd.csv',sep='\t')

    touseX.to_csv(mydata_dir+'/'+resultsfolder+'/touseX_fit.csv',sep='\t')
    touseY.to_csv(mydata_dir+'/'+resultsfolder+'/touseY_fit.csv',sep='\t')

    X_train_un, X_test_un = X_train_pd.values, X_test_pd.values
    Y_train, Y_test = torch.tensor(Y_train_pd.values), torch.tensor(Y_test_pd.values)

    if scaling == "Standardization":
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        X_test = torch.tensor(scaler.transform(X_test_un))

    elif scaling == "Normalization01":

        scaler = MinMaxScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        X_test = torch.tensor(scaler.transform(X_test_un))

    elif scaling == "Normalization-11":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        X_test = torch.tensor(scaler.transform(X_test_un))

    elif scaling == "MaxAbs":
        scaler = MaxAbsScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        X_test = torch.tensor(scaler.transform(X_test_un))

    else:
        scaler = "noscalerused"
        X_train = torch.tensor(X_train_un)
        X_test = torch.tensor(X_test_un)

    model = Network(input_size=input_size, num_hidden_layers=num_hidden_layers,
                    hidden_layers_size=hidden_layers_size,output_size=output_size,
                    activation=activation,dropout_layers=dropout_layers,dropout_rate=dropout_rate,device=device)

    writer.add_graph(model, X_train.float())

    print(model)

    Y_test_word_labels_dict = {}
    for i in Y_test_pd.index.tolist():
        Y_test_word_labels_dict[i] = ",".join(Y_test_pd.columns[Y_test_pd.loc[i,] == 1.0].tolist())

    if model.device == 'cuda':
        model = model.cuda()
        X_train = X_train.to(device=device)
        X_test = X_test.to(device=device)
        Y_train = Y_train.to(device=device)
        Y_test = Y_test.to(device=device)

    helpmat = Y_test.cpu().numpy()

    if weighted_loss == "yes":
        criterion = nn.BCELoss(reduction='none')

    else:
        criterion = nn.BCELoss()

    if optimization == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    elif optimization == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

    outfile = mydata_dir+'/'+resultsfolder+'/init_optim_trained_model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    epoch_loss = []
    validation_loss = []
    epoch_excl_running_loss = []
    epoch_main_running_loss = []
    prec = {key: [] for key in classes}
    rec = {key: [] for key in classes}
    f1 = {key: [] for key in classes}
    acc = {key: [] for key in classes}
    tnr = {key: [] for key in classes}
    npv = {key: [] for key in classes}

    if early_st=='yes':
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=mydata_dir+'/'+resultsfolder+'/early_st_checkpoint.pt')

    stopped_early = "no"
    for epoch in range(num_epochs):

        model.train()

        running_loss = 0
        excl_running_loss = 0
        main_running_loss = 0
        crand = torch.randperm(X_train.shape[0])
        X_train = X_train[crand,]
        Y_train = Y_train[crand,]

        for iterat in range(int(len(X_train)/batch_size)):

            limit1 = iterat*batch_size
            limit2 = (iterat+1)*batch_size

            if limit2 > X_train.shape[0]:
                limit2 = X_train.shape[0]

            toXtrain = X_train[limit1:limit2]
            toYtrain = Y_train[limit1:limit2]
            print(toYtrain)

            optimizer.zero_grad()

            pred_class,outputs = cells_to_probs_class(model, toXtrain, mode="train")

            if weighted_loss == "yes":

                loss = criterion(outputs, toYtrain.float())
                weight_ = np.empty(shape=outputs.shape)

                for pos, mut in enumerate(classes):
                    weight_[:, pos] = [weight[mut][j] for j in toYtrain[:, pos]]

                loss = torch.tensor(weight_).to(device=model.device) * loss
                loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())
                loss_excl = loss_excl.mean()

                main_comp = loss.mean(axis=1).sum().item()
                excl_comp = loss_excl.item() * outputs.shape[0]

                toadd = main_comp + excl_comp
                loss = loss.mean() + loss_excl

            else:

                loss = criterion(outputs, toYtrain.float())
                loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())

                main_comp = loss.item() * outputs.shape[0]
                excl_comp = loss_excl.item() * outputs.shape[0]

                toadd = main_comp + excl_comp
                loss = loss + loss_excl

            loss.backward()

            optimizer.step()

            running_loss = running_loss + toadd
            main_running_loss = main_running_loss + main_comp
            excl_running_loss = excl_running_loss + excl_comp

        epoch_loss.append(running_loss / len(X_train))
        print('[%d] training loss: %.8f' % (epoch, epoch_loss[epoch]))

        epoch_excl_running_loss.append(excl_running_loss / len(X_train))
        epoch_main_running_loss.append(main_running_loss / len(X_train))

        # Set model to test mode
        model.eval()
        pred_class, outputs = cells_to_probs_class(model,X_test,mode="val")

        if weighted_loss == "yes":

            loss = criterion(outputs, Y_test.float())
            weight_ = np.empty(shape=outputs.shape)

            for pos, mut in enumerate(classes):
                weight_[:, pos] = [weight[mut][j] for j in Y_test[:, pos]]

            loss = torch.tensor(weight_).to(device=model.device) * loss

            loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())
            loss_excl = loss_excl.mean()

            loss = loss.mean() + loss_excl

        else:

            loss = criterion(outputs, Y_test.float())

            loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())
            loss = loss + loss_excl

        validation_loss.append(loss.item())
        print('[%d] validation loss: %.8f' % (epoch, loss.item()))

        writer.add_scalars('losses', {'training loss': epoch_loss[epoch],'validation loss': validation_loss[epoch]}, epoch)
        writer.add_scalars('losses_components', {'main loss': epoch_main_running_loss[epoch],
                                                 'exclusive loss': epoch_excl_running_loss[epoch]}, epoch)

        confmat = {}

        for pos,mut in enumerate(classes):

            confmat[pos] = metrics.confusion_matrix(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)),
                                                    np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1)),labels = np.array([0,1]))
            tn, fp, fn, tp = confmat[pos].ravel()
            tnr[mut].append(tnr_fn(fp,tn))
            npv[mut].append(npv_fn(tn,fn))
            prec[mut].append(metrics.precision_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            rec[mut].append(metrics.recall_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            f1[mut].append(metrics.f1_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            acc[mut].append(metrics.accuracy_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))

        if epoch % 20==0:
            for pos,mut in enumerate(classes):

                writer.add_scalars('validation metrics - ' + mut, {'accuracy': acc[mut][epoch],
                                                                   'precision': prec[mut][epoch],
                                                                   'recall': rec[mut][epoch],
                                                                   'NPV': npv[mut][epoch],
                                                                   'TNR': tnr[mut][epoch],
                                                                   'f1': f1[mut][epoch]}, epoch)

            for tag, parm in model.named_parameters():
                writer.add_histogram('Gradient - Validation & '+tag+' & fold', parm.grad.data.cpu().numpy(),epoch)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        if epoch == num_epochs - 1:
            print("HERE2")
            break
        else:
            tune.report(loss=validation_loss[epoch])

        if early_st=='yes':

            early_stopping(validation_loss[epoch], model)

            if early_stopping.early_stop:
                stopped_early = "yes"
                print("Early stopping")
                break

    if stopped_early == 'yes' and epoch != num_epochs-1:
        model.load_state_dict(torch.load(mydata_dir+'/'+resultsfolder+'/early_st_checkpoint.pt'))

        for pos,mut in enumerate(classes):

            tn, fp, fn, tp = confmat[pos].ravel()
            tnr[mut].append(tnr_fn(fp, tn))
            npv[mut].append(npv_fn(tn, fn))
            prec[mut].append(metrics.precision_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            rec[mut].append(metrics.recall_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            f1[mut].append(metrics.f1_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))
            acc[mut].append(metrics.accuracy_score(np.delete(helpmat[:, pos], np.where(helpmat[:, pos] == -1)), np.delete(pred_class[:, pos], np.where(helpmat[:, pos] == -1))))

        if epoch > config['grace_period']:
            epoch = epoch - patience + 1

    outfile = mydata_dir+'/'+resultsfolder+'/metrics_validationFold.json'

    dictmodel={"accuracy": {key: acc[key][epoch] for key in classes},
               "precision": {key: prec[key][epoch] for key in classes},
               "recall": {key: rec[key][epoch] for key in classes},
               "f1": {key: f1[key][epoch] for key in classes},
               "TNR": {key: tnr[key][epoch] for key in classes},
               "NPV": {key: npv[key][epoch] for key in classes},
               "epochs": epoch}

    with open(outfile, 'w') as fp:
        json.dump(dictmodel, fp,sort_keys=True, indent=4)

    print('\n')
    print(dictmodel)
    print("Finished Training")

    writer.close()

    return()


def train_all(config):

    datadict = create_datasets(config)

    touseX = datadict['touseX']
    touseY = datadict['touseY']

    classes = datadict['classes']

    hidden_layers_size = config["hlayers"]
    print("Hidden Layers: %s" % str(hidden_layers_size))

    learning_rate = config["lr"]
    print("Learning rate: %f" % learning_rate)

    batch_size = config["batch_size"]
    print("Batch Size: %d" % batch_size)

    optimization = config["optimization"]
    print("Optimizer: %s" % optimization)

    scaling = config["scaling"]
    print("Input Scaling: %s" % scaling)

    weighted_loss = config["weighted_loss"]
    print("Weighted Loss applied: %s" % weighted_loss)

    weight_decay = config["weight_decay"]
    print("Weight decay for Optimizer: %f" % weight_decay)

    dropout_layers = config["dropout_layers"]
    print("Dropout layers: %s" % str(dropout_layers))

    dropout_rate = config["dropout_rate"]
    print("Dropout rate: %f" % dropout_rate)

    activation = config["activation"]
    print("Activation function: %s" % activation)

    ngenes = config["ngenes"]
    print("Number of genes used in the model: %d" % ngenes)

    mydata_dir = config["data_dir"]
    print("My Results Data Directory: %s" % mydata_dir)

    scripts_dir = config["scripts_dir"]
    print("My scripts Directory: %s" % scripts_dir)

    device = config["device"]
    print("Running on: %s" % device)

    print("\nFitting DNN model...")

    if dropout_layers[-1]>len(hidden_layers_size)-1:
        sys.exit("Dropout layers are more than Hidden Layers. Exiting...")

    num_hidden_layers = len(hidden_layers_size)

    resultsfolder = 'NN_lr'+str(learning_rate)+'_batchsize'+str(batch_size)+'_optimizer'+optimization+'_scaling'+scaling+\
                    '_weightedloss'+weighted_loss+'_weightdecay'+str(weight_decay)+'_activation'+activation+'_dropoutrate'+str(dropout_rate)

    with open(mydata_dir + '/' + resultsfolder + '/metrics_validationFold.json') as json_file:
        num_epochs = json.load(json_file)['epochs']

    print("Epochs: %d" % num_epochs)

    touseX.to_csv(mydata_dir+'/'+resultsfolder+'/touseX.csv',sep='\t')
    touseY.to_csv(mydata_dir+'/'+resultsfolder+'/touseY.csv',sep='\t')

    command='tensorboard --logdir='+mydata_dir+'/'+resultsfolder
    print("\nRun this command '"+command+"' and track the progress of fitting interactively in real time!\n")

    writer = SummaryWriter(mydata_dir+'/'+resultsfolder)

    input_size = len(touseX.columns.tolist())
    output_size = len(classes)

    print(classes)
    weight = {}
    for pos, mut in enumerate(classes):
        if -1. in list(np.unique(touseY.values[:, pos])):
            if (0. in list(np.unique(touseY.values[:, pos]))) and (1. in list(np.unique(touseY.values[:, pos]))):
                weight[mut] = compute_weight_class(touseY.loc[touseY[mut]!=-1,:].values[:, pos], unique_values=[0, 1])
                weight[mut].append(0)
            elif 0. in list(np.unique(touseY.values[:, pos])):
                weight[mut] = [1.0, 0.0, 0.0]
            elif 1. in list(np.unique(touseY.values[:, pos])):
                weight[mut] = [0.0, 1.0, 0.0]
        else:
            weight[mut] = compute_weight_class(touseY.values[:, pos], unique_values=[0, 1])

    X_train_un = touseX.values
    Y_train = torch.tensor(touseY.values)

    if scaling == "Standardization":

        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "Normalization01":

        scaler = MinMaxScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "Normalization-11":

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "MaxAbs":

        scaler = MaxAbsScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un))
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    else:

        scaler = "noscalerused"
        X_train = torch.tensor(X_train_un)

    model = Network(input_size=input_size, num_hidden_layers=num_hidden_layers,
            hidden_layers_size=hidden_layers_size,output_size=output_size,
            activation=activation,dropout_layers=dropout_layers,dropout_rate=dropout_rate,device=device)

    print(model)

    if model.device == 'cuda':
        model = model.cuda()
        X_train = X_train.to(device=device)
        Y_train = Y_train.to(device=device)

    if weighted_loss == "yes":
        criterion = nn.BCELoss(reduction='none')

    else:
        criterion = nn.BCELoss()

    if optimization == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif optimization == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    elif optimization == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif optimization == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    elif(optimization=="Rprop"):
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)

    outfile = mydata_dir+'/'+resultsfolder+'/init_whole_trained_model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

    epoch_loss = []

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0
        crand = torch.randperm(X_train.shape[0])
        X_train = X_train[crand,]
        Y_train = Y_train[crand,]

        for iterat in range(int(len(X_train)/batch_size)):

            limit1 = iterat*batch_size
            limit2 = (iterat+1)*batch_size

            if limit2 > X_train.shape[0]:
                limit2 = X_train.shape[0]

            toXtrain = X_train[limit1:limit2]
            toYtrain = Y_train[limit1:limit2]

            optimizer.zero_grad()

            pred_class,outputs = cells_to_probs_class(model,toXtrain,mode="train")

            if weighted_loss == "yes":

                loss = criterion(outputs, toYtrain.float())
                weight_ = np.empty(shape=outputs.shape)

                for pos, mut in enumerate(classes):
                    weight_[:, pos] = [weight[mut][j] for j in toYtrain[:, pos]]

                loss = torch.tensor(weight_).to(device=model.device) * loss

                loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())
                loss_excl = loss_excl.mean()

                toadd = loss.mean(axis=1).sum().item() + loss_excl.item() * outputs.shape[
                    0]
                loss = loss.mean() + loss_excl

            else:

                loss = criterion(outputs, toYtrain.float())

                loss_excl = criterion(1 - outputs[:, -1], torch.max(outputs[:, :-1], dim=1)[0].clone().detach())

                main_comp = loss.item() * outputs.shape[0]
                excl_comp = loss_excl.item() * outputs.shape[0]

                toadd = main_comp + excl_comp
                loss = loss + loss_excl

            loss.backward()

            optimizer.step()

            running_loss = running_loss + toadd

        epoch_loss.append(running_loss / len(X_train))
        print('[%d] training loss: %.8f' % (epoch, epoch_loss[epoch]))

        writer.add_scalars('Whole Training Dataset', {'training loss': epoch_loss[epoch]}, epoch)

    outfile = mydata_dir+'/'+resultsfolder+'/whole_trained_model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

    print('\nThe saved model is %s' % outfile)

    return model, datadict, writer, scaler, device, mydata_dir, resultsfolder
