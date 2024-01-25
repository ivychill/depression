import argparse
import codecs
import json
import logging.handlers
import os
import random
import time
import numpy as np
import torch
import yaml
from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path
from sklearn.metrics import auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_logger():
    logger = logging.getLogger('voice')
    return logger

logger = get_logger()

def set_logger(logger, log_file=None):
    MAX_LOG_SIZE = 2560000
    LOG_BACKUP_NUM = 4000
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if log_file is not None:
        ff = logging.Formatter(
            '%(asctime)s %(process)d %(processName)s %(filename)s %(lineno)d %(levelname)s %(message)s')
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.info("logger set up")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    return parser


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def macro_mean_roc(fpr,tpr, Neff):
    N = np.shape(fpr)[0]
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    mean_fpr = np.unique(np.concatenate([fpr[i, :Neff[i]] for i in range(N)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(mean_fpr)
    for i in range(N):
        mean_tpr += np.interp(mean_fpr, fpr[i, :Neff[i]], tpr[i, :Neff[i]])
    # Finally average it and compute AUC
    mean_tpr /= N
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc


def plot_roc(fpr,tpr,auc,Method):
    plt.rcParams['savefig.dpi'] = 200  #
    plt.rcParams['figure.dpi'] = 200  #
    plt.figure(1)
    lw = 1.5  # linewidth
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    # plt.plot(fpr, tpr, color='navy', linestyle=':', linewidth=2,
    #          label='AUC of ROC curve (area = %0.2f)' % auc)
    plt.step(fpr, tpr, color='b', lw=lw,
             where='post',label='ROC curve (AUC = %0.4f)' % auc)
    # plt.fill_between(fpr, tpr, step='post', alpha=0.2,
    #                  color='b')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curve_' + str(Method) + '.png')
    plt.show()
    return


def plot_tsne(X_train,y_train,X_test,y_test):
    y_test = 2 + y_test
    y_gt = np.hstack((y_train, y_test))
    data_z= np.vstack((X_train,X_test))
    tsne = TSNE(n_components=2, learning_rate=100)
    tsne.fit_transform(data_z)
    tsne_data = tsne.embedding_
    indx0 = np.where(y_gt == 0)
    indx1 = np.where(y_gt == 1)
    indx2 = np.where(y_gt == 2)
    indx3 = np.where(y_gt == 3)
    plt.rcParams['savefig.dpi'] = 300  #
    plt.rcParams['figure.dpi'] = 300  #
    plt.figure(2)
    s1 = plt.scatter(tsne_data[indx0, 0], tsne_data[indx0, 1], s=40, marker='*', color='black')
    s2 = plt.scatter(tsne_data[indx1, 0], tsne_data[indx1, 1], s=80, marker='.', color='gray')
    s3 = plt.scatter(tsne_data[indx2, 0], tsne_data[indx2, 1], s=40, marker='*', color='b')
    s4 = plt.scatter(tsne_data[indx3, 0], tsne_data[indx3, 1], s=80, marker='.', color='r')
    plt.legend([s1, s2, s3, s4],
               ['$Z_{train}$, label=0', '$Z_{train}$, label=1', '$Z_{test}$, label=0', '$Z_{test}$, label=1'],
               loc='upper right', fontsize=6, edgecolor='gray')
    plt.savefig('T-SNE_Train_Test.png')
    plt.show()
    return


def mini_batch_load_xy(x_all, y_all, batch_size):
    Ndata=x_all.shape[0]
    indx = np.random.permutation(Ndata)
    x_minibatch = x_all[indx[:batch_size], :]
    y_minibatch = y_all[indx[:batch_size]]
    return x_minibatch, y_minibatch


def mini_batch_load_balanced(x_all,y_all,mini_bs,positive_bs):
    Ndata = y_all.shape[0]
    indx = np.random.permutation(Ndata)
    X_trainT = x_all[indx,:]
    y_trainT = y_all[indx]
    X_train = np.zeros_like(X_trainT[0:mini_bs,:])
    y_train = np.zeros_like(y_trainT[0:mini_bs])
    indx1 = np.array(np.where(y_trainT == 1)).squeeze()
    indx0 = np.array(np.where(y_trainT == 0)).squeeze()
    X_train[0:positive_bs,:] = X_trainT[indx1[0:positive_bs],:]
    y_train[0:positive_bs] = y_trainT[indx1[0:positive_bs]]
    X_train[positive_bs:,:] = X_trainT[indx0[0:mini_bs - positive_bs],:]
    y_train[positive_bs:] = y_trainT[indx0[0:mini_bs-positive_bs]]
    # print(f'y_train {y_train, y_train.shape}')
    return X_train, y_train