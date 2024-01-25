
import warnings
from pathlib import Path
import os
import src.configuration as C
import src.models as models
from src.utils import get_parser, load_config, get_logger, set_logger, set_seed
from tqdm import tqdm
from src.dataset import NpyDataset, NpyDatasetContrast
from src.dataset import MatDataset, MatDatasetContrast
from src.criterion import ContrastLoss, CeLoss
import torch
from torch.utils import data
import numpy as np


class Learner():
    def __init__(self):
        self.lambda_ssl = 1

    def train(self):
        model.train()
        # for i, (data_sup, data_ssl) in enumerate(zip(tqdm(dataloader_sup), tqdm(dataloader_ssl))):
        for i, (data_sup) in enumerate(tqdm(dataloader_sup)):
        # for i, (data_ssl) in enumerate(tqdm(dataloader_ssl)):
            image_sup = data_sup['image'].to(device)
            target_sup = data_sup['targets'].to(device)
            output_sup = model(image_sup)
            loss_sup = criterion_sup(output_sup, target_sup)

            # image_ssl = data_ssl['image'].to(device)
            # target_ssl = data_ssl['targets'].to(device)
            # output_ssl = model(image_ssl)
            # loss_ssl = criterion_ssl(output_ssl, target_ssl)

            # loss = loss_sup + loss_ssl * self.lambda_ssl
            loss = loss_sup
            # loss = loss_ssl
            # logger.debug(f'train, epoch {epoch}, step {i}, loss_sup {loss_sup:.4f}, loss_ssl {loss_ssl:.4f}')
            logger.debug(f'train, epoch {epoch}, step {i}, loss_sup {loss_sup:.4f}')
            # logger.debug(f'train, epoch {epoch}, step {i}, loss_ssl {loss_ssl:.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def eval(self):
        model.eval()
        logits_arousals = torch.FloatTensor()
        logits_valances = torch.FloatTensor()
        y_arousals = torch.LongTensor()
        y_valances = torch.LongTensor()

        for i, data_val in enumerate(tqdm(dataloader_val)):
            image = data_val['image'].to(device)
            target = data_val['targets'].to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion_sup(output, target)
            logger.debug(f'eval, epoch {epoch}, step {i}, loss {loss:.4f}')

            logits_arousal = output["arousal"]
            logits_valance = output["valance"]
            y_arousal = target[:, 0]
            y_valance = target[:, 1]

            logits_arousals = torch.cat((logits_arousals, logits_arousal.detach().cpu()), 0)
            logits_valances = torch.cat((logits_valances, logits_valance.detach().cpu()), 0)
            y_arousals = torch.cat((y_arousals, y_arousal.detach().cpu()), 0)
            y_valances = torch.cat((y_valances, y_valance.detach().cpu()), 0)

        _, pred_arousals = torch.max(logits_arousals, 1)
        _, pred_valances = torch.max(logits_valances, 1)
        acc_arousal = (pred_arousals == y_arousals).float().mean()
        acc_valance = (pred_valances == y_valances).float().mean()
        logger.debug(f'eval, epoch {epoch} step {i}, loss {loss:.4f}, acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}')

        return loss, acc_valance

if __name__ == "__main__":
    # target_name = 'arousal'
    target_name = 'valance'
    warnings.filterwarnings("ignore")
    # torch.set_printoptions(threshold=np.inf)
    # np.set_printoptions(threshold=np.inf)

    args = get_parser().parse_args()
    config = load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = get_logger()
    set_logger(logger, output_dir / "train.log")
    set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)

    for i, (trn_idx, val_idx) in enumerate(
            splitter.split(df, y=df[target_name])):
        # if i not in global_params["folds"]:
        #     continue

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        (output_dir/f"fold{i}").mkdir(exist_ok=True, parents=True)
        trn_df.to_csv(output_dir / f"fold{i}"/f'train_{target_name}.csv', index=None)
        val_df.to_csv(output_dir / f"fold{i}"/f'val_{target_name}.csv', index=None)

        # dataset_cls_sup = globals().get('NpyDataset')
        dataset_cls_sup = globals().get('MatDataset')   # 1DCNN
        dataset_sup = dataset_cls_sup(trn_df, datadir)
        dataloader_sup = data.DataLoader(
            dataset_sup, batch_size=32,
            # dataset_sup, batch_size=128,    # 1DCNN
            shuffle=True, num_workers=8, pin_memory=True)

        # dataset_cls_ssl = globals().get('NpyDatasetContrast')
        # # dataset_cls_ssl = globals().get('MatDatasetContrast')    # 1DCNN
        # dataset_ssl = dataset_cls_ssl(df, datadir)
        # dataloader_ssl = data.DataLoader(
        #     dataset_ssl, batch_size=32,
        #     # dataset_ssl, batch_size=128,    # 1DCNN
        #     shuffle=True, num_workers=8, pin_memory=True)

        # dataset_cls_val = globals().get('NpyDataset')
        dataset_cls_val = globals().get('MatDataset')   # 1DCNN
        dataset_val = dataset_cls_val(val_df, datadir)
        dataloader_val = data.DataLoader(
            dataset_val, batch_size=32,
            # dataset_val, batch_size=128,     # 1DCNN
            shuffle=False, num_workers=1, pin_memory=True)

        # logger.debug(f'dataset_sup {len(dataset_sup)}, dataloader_sup {len(dataloader_sup)}, '
        #              f'dataset_ssl {len(dataset_ssl)}, dataloader_ssl {len(dataloader_ssl)}, '
        #              f'dataset_val {len(dataset_val)}, dataloader_val {len(dataloader_val)}')

        model = models.get_model(config).to(device)
        criterion_sup = CeLoss().to(device)
        criterion_ssl = ContrastLoss().to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)

        learner = Learner()
        save_dir = output_dir / f"fold{i}" / 'checkpoints'
        logger.info(f'save_dir {save_dir}')
        save_dir.mkdir(exist_ok=True, parents=True)
        loss_min = np.inf
        acc_max = 0.0
        last_best_epoch = 0
        for epoch in range(global_params["num_epochs"]):
            learner.train()
            loss, acc_valance = learner.eval()
            # if loss < loss_min:
            #     loss_min = loss
            if acc_valance > acc_max:
                acc_max = acc_valance
                logger.debug(f'fold {i}, best epoch {epoch}, loss {loss:.4f}, acc_valance {acc_valance:.4f}')
                try:
                    os.remove(save_dir/f'{last_best_epoch}.pth')
                except:
                    logger.error(f"Error while deleting file {last_best_epoch}.pth")
                torch.save(model.state_dict(), save_dir/f'{epoch}.pth')
                torch.save(model.state_dict(), save_dir/'best.pth')
                last_best_epoch = epoch
            # if epoch % 200 == 199:
            #     torch.save(model.state_dict(), save_dir/f'{epoch}.pth')
        torch.save(model.state_dict(), save_dir/'last.pth')