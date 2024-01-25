
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import scipy.io
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pylab
from src.utils import get_logger, set_logger, set_seed


class FeatureDataset(data.Dataset):
    def __init__(self, feat_path):
        result = scipy.io.loadmat(feat_path)
        self.feature = torch.from_numpy(result['feature'])
        self.arousal = torch.from_numpy(result['arousal']).squeeze(0)
        self.valance = torch.from_numpy(result['valance']).squeeze(0)
        # logger.debug(f'feat_path {feat_path}')
        # logger.debug(f'arousal {self.arousal}')
        # logger.debug(f'valance {self.valance}')

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index: int):
        # print(f'type {self.arousal[index].dtype, self.valance[index].dtype}')
        return self.feature[index], self.arousal[index].long(), self.valance[index].long()


class Classifer(nn.Module):
    # def __init__(self, dim_feature=256,
    def __init__(self, dim_feature=16,    # 1DCNN
                 dim_arousal=3,
                 dim_valance=3):
        super(Classifer, self).__init__()
        dim_latent = 32
        self.Linear_layer = nn.Linear(dim_feature, dim_latent)
        self.Activate = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.ReLU(True),
        )
        self.classifier_arousal = nn.Linear(dim_latent, dim_arousal)
        self.classifier_valance = nn.Linear(dim_latent, dim_valance)

    def forward(self, x):
        x = self.Linear_layer(x)
        z = self.Activate(x)
        arousal = self.classifier_arousal(z)
        valance = self.classifier_valance(z)

        return arousal, valance

class Learner():
    def __init__(self):
        pass

    def train(self):
        model.train()
        pred_arousals = torch.LongTensor()
        pred_valances = torch.LongTensor()
        y_arousals = torch.LongTensor()
        y_valances = torch.LongTensor()

        for i, (feature, y_arousal, y_valance) in enumerate(tqdm(train_loader)):
            feature = feature.to(device)
            # feature = F.normalize(feature)
            y_arousal = y_arousal.to(device)
            y_valance = y_valance.to(device)
            logits_arousal, logits_valance = model(feature)
            loss_arousal = criterion(logits_arousal, y_arousal)
            loss_valance = criterion(logits_valance, y_valance)
            loss = loss_arousal + loss_valance
            # logger.debug(
                # f'train, epoch {epoch}, step {i}, loss_arousal {loss_arousal:.4f}, loss_valance {loss_valance:.4f}')

            _, pred_arousal = torch.max(logits_arousal, 1)
            _, pred_valance = torch.max(logits_valance, 1)
            pred_arousals = torch.cat((pred_arousals, pred_arousal.detach().cpu()), 0)
            pred_valances = torch.cat((pred_valances, pred_valance.detach().cpu()), 0)
            y_arousals = torch.cat((y_arousals, y_arousal.detach().cpu()), 0)
            y_valances = torch.cat((y_valances, y_valance.detach().cpu()), 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_arousal = (pred_arousals == y_arousals).float().mean()
        acc_valance = (pred_valances == y_valances).float().mean()
        logger.debug(f'train, epoch {epoch} acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}')

    def eval(self):
        model.eval()

        logits_arousals = torch.FloatTensor()
        logits_valances = torch.FloatTensor()
        pred_arousals = torch.LongTensor()
        pred_valances = torch.LongTensor()
        y_arousals = torch.LongTensor()
        y_valances = torch.LongTensor()

        for i, (feature, y_arousal, y_valance) in enumerate(tqdm(val_loader)):
            feature = feature.to(device)
            y_arousal = y_arousal.to(device)
            y_valance = y_valance.to(device)
            logits_arousal, logits_valance = model(feature)
            loss_arousal = criterion(logits_arousal, y_arousal)
            loss_valance = criterion(logits_valance, y_valance)
            _, pred_arousal = torch.max(logits_arousal, 1)
            _, pred_valance = torch.max(logits_valance, 1)
            # logger.debug(f'eval, loss_arousal {loss_arousal:.4f}, loss_valance {loss_valance:.4f}')
            logits_arousals = torch.cat((logits_arousals, logits_arousal.detach().cpu()), 0)
            logits_valances = torch.cat((logits_valances, logits_valance.detach().cpu()), 0)
            pred_arousals = torch.cat((pred_arousals, pred_arousal.detach().cpu()), 0)
            pred_valances = torch.cat((pred_valances, pred_valance.detach().cpu()), 0)
            y_arousals = torch.cat((y_arousals, y_arousal.detach().cpu()), 0)
            y_valances = torch.cat((y_valances, y_valance.detach().cpu()), 0)

        acc_arousal = (pred_arousals == y_arousals).float().mean()
        acc_valance = (pred_valances == y_valances).float().mean()
        # logger.debug(f'eval, epoch {epoch} acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}')
        return logits_arousals, logits_valances, y_arousals, y_valances, acc_arousal, acc_valance


def cal_auc(logits, y, name):
    softmax = nn.Softmax(dim=1)
    prob = softmax(logits).numpy()
    class_num = 3
    y_one_hot = label_binarize(y.numpy(), np.arange(class_num))
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), prob.ravel())
    auc = metrics.auc(fpr, tpr)
    return auc
    # plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.4f' % auc)
    # plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    # plt.xlim((-0.01, 1.02))
    # plt.ylim((-0.01, 1.02))
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('False Positive Rate', fontsize=13)
    # plt.ylabel('True Positive Rate', fontsize=13)
    # plt.grid(b=True, ls=':')
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.title(f'{name} ROC and AUC', fontsize=17)
    #
    # fig = pylab.gcf()
    # fig.canvas.set_window_title(name)
    #
    # plt.show()


if __name__ == "__main__":
    dataset = 'baidu'
    # dataset = 'IEMOCAP'
    # name = 'arousal'
    name = 'valance'

    torch.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=np.inf)
    base_dir = Path(f'../user_data/{dataset}/1d_my')
    logger = get_logger()
    set_logger(logger, base_dir / "test.log")
    set_seed(1213)
    device = torch.device('cuda')

    model = Classifer().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc_arousals = []
    acc_valances = []
    auc_arousals = []
    auc_valances = []
    fold_num = 5
    for index in range(fold_num):
        logger.debug(f'{name} pretext fold {index}')
        train_dataset = FeatureDataset(base_dir/f'fold{index}'/f'train_{name}.mat')
        train_loader = data.DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=8, pin_memory=True)

        val_dataset = FeatureDataset(base_dir/f'fold{index}'/f'val_{name}.mat')
        val_loader = data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=1, pin_memory=True)

        learner = Learner()

        n_epoch = 10
        for epoch in range(n_epoch):
            learner.train()
            logits_arousals, logits_valances, y_arousals, y_valances, acc_arousal, acc_valance = learner.eval()

        auc_arousal = cal_auc(logits_arousals, y_arousals, name)
        auc_valance = cal_auc(logits_valances, y_valances, name)

        logger.debug(f'fold {index}, '
                     f'acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}, '
                     f'auc_arousal {auc_arousal:.4f}, auc_valance {auc_valance:.4f}')

        acc_arousals.append(acc_arousal)
        acc_valances.append(acc_valance)
        auc_arousals.append(auc_arousal)
        auc_valances.append(auc_valance)

    acc_arousals = np.mean(np.array(acc_arousals))
    acc_valances = np.mean(np.array(acc_valances))
    auc_arousals = np.mean(np.array(auc_arousals))
    auc_valances = np.mean(np.array(auc_valances))
    logger.debug(f'acc_arousals {acc_arousals:.4f}, acc_valances {acc_valances:.4f}, '
                 f'auc_arousals {auc_arousals:.4f}, auc_valances {auc_valances:.4f}')