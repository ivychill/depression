
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms
from pathlib import Path
import numpy as np
# from sklearn.preprocessing import StandardScaler
from src.dataset import *
from src.criterion import *


def get_device(device: str):
    return torch.device(device)


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get("params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    return sms.__getattribute__(name)(**split_config["params"])


def replaceNaNWithMean(datMat):
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i]))[0],i])
        datMat[np.nonzero(np.isnan(datMat[:,i]))[0],i] = meanVal
    return datMat

def get_metadata(config: dict):
    data_config = config["data"]
    meta_data = pd.read_csv(data_config["train_df_path"])
    data_dir = Path(data_config["train_audio_path"])
    return meta_data, data_dir


def get_loader(df: pd.DataFrame, datadir: Path, config: dict, phase: str):
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    if hasattr(nn, dataset_name):
        dataset = nn.__getattribute__(dataset_name)(df, datadir)
    else:
        dataset_cls = globals().get(dataset_name)
        if dataset_cls is not None:
            dataset = dataset_cls(df, datadir)
        else:
            raise NotImplementedError

    # if dataset_config["name"] == "NpyDataset":
    #     dataset = datasets.NpyDataset(df, datadir)
    # elif dataset_config["name"] == "TestNpyDataset":
    #     dataset = datasets.TestNpyDataset(df, datadir)
    # elif dataset_config["name"] == "NpyDatasetJigsaw":
    #     dataset = datasets.NpyDatasetJigsaw(df, datadir)
    # elif dataset_config["name"] == "NpyDatasetContrast":
    #     dataset = datasets.NpyDatasetContrast(df, datadir)
    # elif dataset_config["name"] == "TestNpyDatasetContrast":
    #     dataset = datasets.TestNpyDatasetContrast(df, datadir)
    # elif dataset_config["name"] == "MatDataset":
    #     dataset = datasets.MatDataset(df, datadir)
    # elif dataset_config["name"] == "MatDatasetContrast":
    #     dataset = datasets.MatDatasetContrast(df, datadir)
    # else:
    #     raise NotImplementedError

    loader_config = config["loader"][phase]
    loader = data.DataLoader(dataset, **loader_config)
    return loader