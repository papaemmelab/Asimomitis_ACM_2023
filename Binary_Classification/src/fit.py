from Network import *
import shutil
from es import *
import numpy as np
from utils_Network import *
from prepare import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import metrics
from torch import nn
import torch
import json
from torch.utils.tensorboard import SummaryWriter
import os
from ray import tune
import sys
from pickle import dump


def fit(config, checkpoint_dir=None, data_dir=None):

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
    print("Weighted Loss for WT: %s" % weighted_loss)

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

    device = config["device"]
    print("Running on: %s" % device)

    print("\nFitting DNN model...")

    if (dropout_layers[-1]>len(hidden_layers_size)-1):
        sys.exit("Dropout layers are more than Hidden Layers. Exiting...")

    num_hidden_layers = len(hidden_layers_size)

    resultsfolder = 'NN_lr'+str(learning_rate)+'_batchsize'+str(batch_size)+'_optimizer'+optimization+'_scaling'+scaling+\
                    '_weightedloss'+weighted_loss+'_weightdecay'+str(weight_decay)+'_activation'+activation+'_dropoutrate'+str(dropout_rate)

    print(os.getcwd())

    if os.path.isdir(mydata_dir+'/'+resultsfolder):
        dir_path = mydata_dir+'/'+resultsfolder
        shutil.rmtree(dir_path)

    command='tensorboard --logdir='+mydata_dir+'/'+resultsfolder
    print("\nRun this command '"+command+"' and track the progress of fitting interactively in real time!\n")

    writer = SummaryWriter(mydata_dir+'/'+resultsfolder)

    if weighted_loss == "yes":
        weight = compute_weight_class(touseY.values)
        print(weight)

    input_size = len(touseX.columns.tolist())
    output_size = 1

    X_train_pd, X_test_pd, Y_train_pd, Y_test_pd = train_test_split(touseX, touseY, stratify=touseY, test_size=0.2,random_state=42)

    X_train_pd.to_csv(mydata_dir+'/'+resultsfolder+'/X_train_pd.csv',sep='\t')
    Y_train_pd.to_csv(mydata_dir+'/'+resultsfolder+'/Y_train_pd.csv',sep='\t')

    X_test_pd.to_csv(mydata_dir+'/'+resultsfolder+'/X_test_pd.csv',sep='\t')
    Y_test_pd.to_csv(mydata_dir+'/'+resultsfolder+'/Y_test_pd.csv',sep='\t')

    touseX.to_csv(mydata_dir+'/'+resultsfolder+'/touseX_fit.csv',sep='\t')
    touseY.to_csv(mydata_dir+'/'+resultsfolder+'/touseY_fit.csv',sep='\t')

    X_train_un, X_test_un = X_train_pd.values, X_test_pd.values
    Y_train, Y_test = torch.tensor(Y_train_pd.values), torch.tensor(Y_test_pd.values)

    if scaling=="Standardization":

        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        X_test = torch.tensor(scaler.transform(X_test_un)).to(device=device)

    elif scaling=="Normalization01":

        scaler = MinMaxScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        X_test = torch.tensor(scaler.transform(X_test_un)).to(device=device)

    elif scaling=="Normalization-11":

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        X_test = torch.tensor(scaler.transform(X_test_un)).to(device=device)

    elif scaling=="MaxAbs":

        scaler = MaxAbsScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        X_test = torch.tensor(scaler.transform(X_test_un)).to(device=device)

    else:

        scaler = "noscalerused"
        X_train = torch.tensor(X_train_un).to(device=device)
        X_test = torch.tensor(X_test_un).to(device=device)

    if (np.where(Y_test.numpy()==1)[0].shape[0] == Y_test.numpy().shape[0]):
        raise Exception("No WT cells in the validation set")

    if np.where(Y_train.numpy()==1)[0].shape[0] == Y_train.numpy().shape[0]:
        raise Exception("No WT cells in the training set")

    model = Network(input_size=input_size, num_hidden_layers=num_hidden_layers,
                    hidden_layers_size=hidden_layers_size,output_size=output_size,
                    activation=activation,dropout_layers=dropout_layers,dropout_rate=dropout_rate,device=device)

    writer.add_graph(model, X_train.float())

    print(model)

    if model.device == 'cuda':
        model = model.cuda()

        Y_train = Y_train.to(device=device)
        Y_test = Y_test.to(device=device)

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
    prec = []
    rec = []
    f1 = []
    acc = []
    tnr = []
    npv = []

    if early_st=='yes':
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=mydata_dir+'/'+resultsfolder+'/early_st_checkpoint.pt')

    stopped_early = "no"

    for epoch in range(num_epochs):

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

            model.train()

            optimizer.zero_grad()

            pred_class,outputs = cells_to_probs_class(model,toXtrain,mode="train")

            temp = toYtrain.shape[0]

            if (weighted_loss == "yes"):

                loss = criterion(outputs, toYtrain.float().view(temp, 1))
                weight_ = [weight[j] for j in torch.transpose(toYtrain, 1, 0).to(dtype=torch.int32)[0]]
                loss = torch.tensor(np.array(weight_).reshape(-1, 1)).to(device=model.device) * loss
                loss = loss.mean()

            else:

                loss = criterion(outputs, toYtrain.float().view(temp,1))

            loss.backward()

            optimizer.step()

            running_loss = running_loss + loss.item()*toXtrain.shape[0]

        epoch_loss.append(running_loss / len(X_train))
        print('[%d] training loss: %.8f' % (epoch, epoch_loss[epoch]))

        model.eval()

        pred_class, outputs = cells_to_probs_class(model,X_test,mode="val")
        temp = Y_test.shape[0]

        if (weighted_loss == "yes"):

            loss = criterion(outputs, Y_test.float().view(temp, 1))
            weight_ = [weight[j] for j in torch.transpose(Y_test, 1, 0).to(dtype=torch.int32)[0]]
            loss = torch.tensor(np.array(weight_).reshape(-1, 1)).to(device=model.device) * loss
            loss = loss.mean()

        else:

            loss = criterion(outputs, Y_test.float().view(temp, 1))

        validation_loss.append(loss.item())
        print('[%d] validation loss: %.8f' % (epoch, loss.item()))

        writer.add_scalars('losses - fold ', {'training loss': epoch_loss[epoch],'validation loss': validation_loss[epoch]}, epoch)

        helpmat = Y_test.cpu().numpy()
        prec.append(metrics.precision_score(helpmat, pred_class))
        rec.append(metrics.recall_score(helpmat, pred_class))
        f1.append(metrics.f1_score(helpmat, pred_class))
        acc.append(metrics.accuracy_score(helpmat, pred_class))

        confmat = metrics.confusion_matrix(helpmat, pred_class)
        tn, fp, fn, tp = confmat.ravel()
        if (tn+fp)!=0:
            tnr.append(tn / (tn+fp))
        else:
            tnr.append(np.nan)

        if (tn+fn)!=0:
            npv.append(tn / (tn+fn))
        else:
            npv.append(np.nan)

        writer.add_scalars('validation metrics - fold ', {'accuracy': acc[epoch],
                                                          'precision': prec[epoch],
                                                          'recall': rec[epoch],
                                                          'NPV':npv[epoch],
                                                          'TNR':tnr[epoch],
                                                          'f1': f1[epoch]}, epoch)

        if epoch % 20 == 0:

            for tag, parm in model.named_parameters():
                writer.add_histogram('Gradient - Validation & '+tag+' & fold',parm.grad.data.cpu().numpy(),epoch)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=validation_loss[epoch], accuracy=acc[epoch], precision=prec[epoch], recall=rec[epoch], NPV=npv[epoch], TNR=tnr[epoch], f1=f1[epoch])

        if early_st=='yes':

            early_stopping(validation_loss[epoch], model)

            if early_stopping.early_stop:
                stopped_early = "yes"
                print("Early stopping")
                break

    if stopped_early == 'yes' and epoch != num_epochs-1:
        model.load_state_dict(torch.load(mydata_dir+'/'+resultsfolder+'/early_st_checkpoint.pt'))

        if epoch > config['grace_period']:
            epoch = epoch - patience + 1

    outfile = mydata_dir+'/'+resultsfolder+'/metrics_validationFold.json'

    dictmodel={"accuracy": acc[epoch],
               "precision": prec[epoch],
               "recall": rec[epoch],
               "f1": f1[epoch],
               "TNR":tnr[epoch],
               "NPV":npv[epoch],
               "epochs":epoch}

    with open(outfile, 'w') as fp:
        json.dump(dictmodel, fp,sort_keys=True, indent=4)

    print('\n')
    print(dictmodel)
    print("Finished Training")

    return()


def train_all(config,num_epochs):

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
    print("Weighted Loss for WT: %s" % weighted_loss)

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

    if os.path.exists(mydata_dir + '/' + resultsfolder + '/metrics_validationFold.json'):
        with open(mydata_dir + '/' + resultsfolder + '/metrics_validationFold.json') as json_file:
            num_epochs = json.load(json_file)['epochs']

    print("Epochs: %d" % num_epochs)

    touseX.to_csv(mydata_dir+'/'+resultsfolder+'/touseX.csv',sep='\t')
    touseY.to_csv(mydata_dir+'/'+resultsfolder+'/touseY.csv',sep='\t')

    command='tensorboard --logdir='+mydata_dir+'/'+resultsfolder
    print("\nRun this command '"+command+"' and track the progress of fitting interactively in real time!\n")

    writer = SummaryWriter(mydata_dir+'/'+resultsfolder)

    if weighted_loss=="yes":
        weight = compute_weight_class(touseY.values)
        print(weight)

    input_size = len(touseX.columns.tolist())
    output_size = 1

    X_train_un = touseX.values
    Y_train = torch.tensor(touseY.values)

    if scaling == "Standardization":

        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "Normalization01":

        scaler = MinMaxScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "Normalization-11":

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    elif scaling == "MaxAbs":

        scaler = MaxAbsScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train_un)).to(device=device)
        outfile = mydata_dir+'/'+resultsfolder + '/scaler.pkl'
        dump(scaler, open(outfile, 'wb'))

    else:

        scaler="noscalerused"
        X_train = torch.tensor(X_train_un).to(device=device)

    model = Network(input_size=input_size,num_hidden_layers=num_hidden_layers,
                    hidden_layers_size=hidden_layers_size,output_size=output_size,
                    activation=activation,dropout_layers=dropout_layers,dropout_rate=dropout_rate,device=device)

    print(model)

    if model.device == 'cuda':
        model = model.cuda()
        Y_train = Y_train.to(device=device)

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

    outfile = mydata_dir+'/'+resultsfolder+'/init_whole_trained_model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

    epoch_loss = []

    for epoch in range(num_epochs):

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

            model.train()

            optimizer.zero_grad()

            pred_class,outputs = cells_to_probs_class(model,toXtrain,mode="train")

            temp = toYtrain.shape[0]

            if weighted_loss == "yes":

                loss = criterion(outputs, toYtrain.float().view(temp, 1))
                weight_ = [weight[j] for j in torch.transpose(toYtrain, 1, 0).to(dtype=torch.int32)[0]]
                loss = torch.tensor(np.array(weight_).reshape(-1, 1)).to(device=model.device) * loss
                loss = loss.mean()

            else:

                loss = criterion(outputs, toYtrain.float().view(temp,1))

            loss.backward()

            optimizer.step()

            running_loss = running_loss + loss.item()*toXtrain.shape[0]

        epoch_loss.append(running_loss / len(X_train))
        print('[%d] training loss: %.8f' % (epoch, epoch_loss[epoch]))

        writer.add_scalars('Whole Training Dataset', {'training loss': epoch_loss[epoch]}, epoch)

    outfile = mydata_dir+'/'+resultsfolder+'/whole_trained_model.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

    print('\nThe saved model is %s' % outfile)

    return model,datadict,writer,scaler,device,mydata_dir,resultsfolder


