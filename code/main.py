# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from log import get_logger, set_logger
from dataset import NpyDataset
from models import ResNet
from criterion import CeLoss



def test(model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    probs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target[0], target[1], target[2], target[3] = \
                data.to(device), target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device)
            output = model(data)
            logits_id, logits_gender, age_pred, logits_psy = output[0], output[1], output[2], output[3]
            id, gender, age, psy = target[0], target[1], target[2], target[3]
            prob = F.softmax(logits_psy, dim=1)
            # print(target.shape, prob.shape)
            targets.append(psy)
            probs.append(prob)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target, reduction='sum').item()
            pred = logits_psy.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(psy.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    targets = torch.hstack(targets)
    probs = torch.vstack(probs)[:,1]    # todo, [:,1] applicable to binary classification
    # print(targets.shape, probs.shape)
    auc = roc_auc_score(targets.cpu(), probs.cpu(), multi_class='ovr')
    
    logger.debug('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.4f}'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_accuracy, auc))
    
    return test_loss, test_accuracy, auc


def train_valid_test_split(df, target, n_splits=5, valid_fold=0, test_fold=1):
    """
    Splits a pandas DataFrame into train, validation, and test sets using stratified k-fold cross-validation.

    Parameters:
    df (pandas DataFrame): The DataFrame to split.
    target (str): The name of the target column.
    n_splits (int): The number of folds to use for cross-validation. Default is 5.
    valid_fold (int): The fold number to use for validation. Default is 0.
    test_fold (int): The fold number to use for testing. Default is 1.

    Returns:
    tuple: A tuple containing the X and y values for the train, validation, and test sets.

    """
    # Initialize StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Generate fold indices and assign fold numbers to each row in DataFrame
    df['kfold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df[target].values)):
        df.loc[val_idx, 'kfold'] = fold

    # Create DataFrames for train, validation, and test sets
    df_valid = df[df.kfold == valid_fold].reset_index(drop=True)
    df_test = df[df.kfold == test_fold].reset_index(drop=True)
    df_train = df[df.kfold != valid_fold]
    df_train = df_train[df_train.kfold != test_fold].reset_index(drop=True)

    # Drop kfold column from each DataFrame
    df_train = df_train.drop("kfold", axis=1)
    df_valid = df_valid.drop("kfold", axis=1)
    df_test = df_test.drop("kfold", axis=1)

    # Split each DataFrame into X and y values
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_valid = df_valid.drop(target, axis=1)
    y_valid = df_valid[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    dim_id = y_train.count()
    dim_psy  = y_train.value_counts().count()
    
    # Print out shape and distribution of target variable for each set
    logger.debug(f"df shape was: {df.shape}")
    logger.debug(f"X_train shape: {X_train.shape}")
    logger.debug(f"X_valid shape: {X_valid.shape}")
    logger.debug(f"X_test shape: {X_test.shape}")
    logger.debug(f"y_train distribution:\n {y_train.value_counts()}")  
    logger.debug(f"y_valid distribution:\n {y_valid.value_counts()}")
    logger.debug(f"y_test distribution: \n {y_test.value_counts()}")
    logger.debug(f'dim_id dim {dim_id}, dim_psy dim {dim_psy}')

    # # Return X and y values for each set
    # return X_train, y_train, X_valid, y_valid, X_test, y_test
    return df_train, df_valid, df_test, dim_id, dim_psy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')      # todo: epoch
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=16, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(0)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': 64, 'shuffle': True}
    test_kwargs = {'batch_size': 64, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    data_root = Path('../user_data/')
    df = pd.read_excel(data_root/'desc.xlsx')   # todo
    df = df[df['危机评估'].notnull()].reset_index(drop=True)    # 90, thereof 0:59, 1:27, 2:4

    train_accuracy_list = []
    test_accuracy_list = []
    test_auc_list = []
    for run in range(10):
        logger.debug(f'run {run}')
        best_loss = 1.0
        best_accuracy = 0
        test_loss = 1.0
        test_accuracy = 0
        df_train, df_valid, df_test, dim_id, dim_psy = train_valid_test_split(df, target='危机评估')
        dim_psy = 2 # todo
        train_dataset = NpyDataset(df_train, data_root)
        valid_dataset = NpyDataset(df_valid, data_root)
        test_dataset = NpyDataset(df_test, data_root)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, **test_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        
        model = ResNet(base_model_name='resnet34', dim_id=dim_id, dim_psy=dim_psy, pretrained=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        criterion = CeLoss()
        
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                data, target[0], target[1], target[2], target[3] = \
                    data.to(device), target[0].to(device), target[1].to(device), target[2].to(device), target[3].to(device)
                optimizer.zero_grad()
                output = model(data)
                # loss = F.nll_loss(output, target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if ((batch_idx + 1) % args.log_interval == 0) or ((batch_idx + 1) == len(train_loader)):
                    logits_id, logits_gender, age_pred, logits_psy = output[0], output[1], output[2], output[3]
                    id, gender, age, psy = target[0], target[1], target[2], target[3]
                    pred = logits_psy.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct = pred.eq(psy.view_as(pred)).sum().item()
                    accuracy = correct / len(data)
                    logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        epoch, (batch_idx + 1)  * len(data), len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), loss.item(), 
                        correct, len(data), 100. * accuracy))
                    valid_loss, valid_accuracy, valid_auc = test(model, criterion, device, valid_loader)
                    if valid_accuracy >= best_accuracy:
                        best_loss = valid_loss
                        best_accuracy = valid_accuracy
                        test_loss, test_accuracy, test_auc = test(model, criterion, device, test_loader)
                        train_loss = loss.item()
                        train_accuracy = accuracy
                        
                        logger.debug(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Valid AUC: {valid_auc:.4f}')
                        logger.debug(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}')
                        if args.save_model:
                            torch.save(model.state_dict(), model_dir/"depression.pt")  # todo
                    
            scheduler.step()
        
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        test_auc_list.append(test_auc)

    train_accuracy_mean = np.mean(train_accuracy_list)
    test_accuracy_mean = np.mean(test_accuracy_list)
    test_auc_mean = np.mean(test_auc_list)
    logger.debug(f'train_accuracy_list: {train_accuracy_list}, mean {train_accuracy_mean:.4f}')
    logger.debug(f'test_accuracy_list: {test_accuracy_list}, mean {test_accuracy_mean:.4f}')
    logger.debug(f'test_auc_list: {test_auc_list}, mean {test_auc_mean:.4f}')   


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    logger = get_logger()
    log_dir = Path('../log')
    log_dir.mkdir(parents=True, exist_ok=True)
    set_logger(logger, log_dir / "dpr.log") # todo
    
    model_dir = Path('../model')
    model_dir.mkdir(parents=True, exist_ok=True)
    main()