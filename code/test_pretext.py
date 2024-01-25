
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import scipy.io
from pathlib import Path
from tqdm import tqdm
from src.utils import get_parser, load_config, get_logger, set_logger, set_seed


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
    def __init__(self, dim_feature=256,
    # def __init__(self, dim_feature=16,    # 1DCNN
                 dim_arousal=3,
                 # dim_valance=3):
                 dim_valance=2):
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
        # pred_arousals = torch.LongTensor()
        # pred_valances = torch.LongTensor()
        y_arousals = torch.LongTensor()
        y_valances = torch.LongTensor()

        for i, (feature, y_arousal, y_valance) in enumerate(tqdm(val_loader)):
            feature = feature.to(device)
            y_arousal = y_arousal.to(device)
            y_valance = y_valance.to(device)
            logits_arousal, logits_valance = model(feature)
            # loss_arousal = criterion(logits_arousal, y_arousal)
            # loss_valance = criterion(logits_valance, y_valance)
            # logger.debug(f'eval, loss_arousal {loss_arousal:.4f}, loss_valance {loss_valance:.4f}')
            logits_arousals = torch.cat((logits_arousals, logits_arousal.detach().cpu()), 0)
            logits_valances = torch.cat((logits_valances, logits_valance.detach().cpu()), 0)
            y_arousals = torch.cat((y_arousals, y_arousal.detach().cpu()), 0)
            y_valances = torch.cat((y_valances, y_valance.detach().cpu()), 0)

        _, pred_arousals = torch.max(logits_arousals, 1)
        _, pred_valances = torch.max(logits_valances, 1)
        acc_arousal = (pred_arousals == y_arousals).float().mean()
        acc_valance = (pred_valances == y_valances).float().mean()
        logger.debug(f'eval, epoch {epoch} acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}')
        return acc_valance

if __name__ == "__main__":
    data_source = 'baidu'
    # data_source = 'IEMOCAP'
    # name = 'arousal'
    name = 'valance'

    torch.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=np.inf)

    args = get_parser().parse_args()
    config = load_config(args.config)
    global_params = config["globals"]
    output_dir = Path(global_params["output_dir"])

    logger = get_logger()
    set_logger(logger, output_dir / "test_pretext.log")
    set_seed(1213)
    device = torch.device('cuda')

    model = Classifer().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fold_num = 5
    for index in range(fold_num):
        base_dir = output_dir/f'fold{index}'
        logger.info(f'base_dir {base_dir}')
        train_dataset = FeatureDataset(base_dir/f'train_{name}.mat')
        train_loader = data.DataLoader(
            train_dataset, batch_size=128, shuffle=True,
            num_workers=8, pin_memory=True)

        val_dataset = FeatureDataset(base_dir/f'val_{name}.mat')
        val_loader = data.DataLoader(
            val_dataset, batch_size=128, shuffle=False,
            num_workers=1, pin_memory=True)

        learner = Learner()
        save_dir = base_dir/'checkpoints'
        logger.info(f'save_dir {save_dir}')
        save_dir.mkdir(exist_ok=True, parents=True)
        acc_max = 0.0
        last_best_epoch = 0
        n_epoch = 10
        for epoch in range(n_epoch):
            learner.train()
            acc_valance = learner.eval()
            if acc_valance > acc_max:
                acc_max = acc_valance
                logger.debug(f'fold {index}, best epoch {epoch}, acc_valance {acc_valance:.4f}')
                try:
                    os.remove(save_dir/f'classifier_{last_best_epoch}.pth')
                except:
                    logger.error(f"Error while deleting file {last_best_epoch}.pth")
                torch.save(model.state_dict(), save_dir/f'classifier_{epoch}.pth')
                torch.save(model.state_dict(), save_dir/'classifier_best.pth')
                last_best_epoch = epoch

        torch.save(model.state_dict(), save_dir / 'classifier_last.pth')