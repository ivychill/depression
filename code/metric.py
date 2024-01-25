
import warnings
import src.configuration as C
import src.models as models
from src.utils import get_parser, load_config, get_logger, set_logger, set_seed
import torch
from torch import nn
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.dataset import EnsembleNpyDataset, EnsembleMatDataset
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def predict_proba(logits):
    softmax = nn.Softmax(dim=1)
    pred = softmax(logits)
    return pred


if __name__ == "__main__":
    data_source = 'baidu'
    target_name = 'valance'
    
    np.set_printoptions(threshold=np.inf)
    warnings.filterwarnings("ignore")
    args = get_parser().parse_args()
    config = load_config(args.config)
    global_params = config["globals"]
    ensemble = bool(global_params['ensemble'])
    output_dir = Path(global_params["output_dir"])

    logger = get_logger()
    set_logger(logger, output_dir / "metric.log")
    set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])
    model = models.get_model(config).to(device)

    for subset in ['train', 'val']:
        proba_arousals = torch.FloatTensor()
        proba_valances = torch.FloatTensor()
        y_arousals = torch.LongTensor()
        y_valances = torch.LongTensor()

        fold_num = 5
        for index in range(fold_num):
            base_dir = output_dir/f'fold{index}'
            weights_path = base_dir/'checkpoints'/'best.pth'
            logger.info(f'weights_path {weights_path}')
    
            checkpoint = torch.load(weights_path)
            if "model_state_dict" in checkpoint:
                logger.info(f'catalyst framework')
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.info(f'my own framework')
                model.load_state_dict(checkpoint)
            model.eval()
    
            path = base_dir/f'{subset}_{target_name}.csv'
            logger.info(f'path {path}')
            meta_data = pd.read_csv(path)
            data_dir = Path(config["data"]["train_audio_path"])
            if ensemble:
                dataset = EnsembleNpyDataset(meta_data, data_dir)
                # dataset = EnsembleMatDataset(meta_data, data_dir)   # 1DCNN
                loader = data.DataLoader(
                    dataset, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True) # batch_size should be 1
            else:
                loader = C.get_loader(meta_data, data_dir, config, "valid")

            for input in tqdm(loader):
                image_array = input['image'].to(device)
                if ensemble:
                # if len(image_array.size()) == 5:    # EnsembleMatDataset
                    images = image_array.squeeze(0)
                else:
                    images = image_array
                labels = input['targets'].to(device)
                with torch.no_grad():
                    output = model(images)
                logits_arousal = output['arousal']
                logits_valance = output['valance']
                proba_arousal = predict_proba(logits_arousal)
                proba_valance = predict_proba(logits_valance)
                if ensemble:
                # if len(image_array.size()) == 5:    # EnsembleMatDataset
                    proba_arousal = torch.mean(proba_arousal, 0)
                    proba_valance = torch.mean(proba_valance, 0)
                    proba_arousal = torch.unsqueeze(proba_arousal, 0)
                    proba_valance = torch.unsqueeze(proba_valance, 0)
                proba_arousals = torch.cat((proba_arousals, proba_arousal.detach().cpu()), 0)
                proba_valances = torch.cat((proba_valances, proba_valance.detach().cpu()), 0)
                y_arousal = labels[:, 0]
                y_valance = labels[:, 1]
                y_arousals = torch.cat((y_arousals, y_arousal.detach().cpu()), 0)
                y_valances = torch.cat((y_valances, y_valance.detach().cpu()), 0)

        _, pred_arousals = torch.max(proba_arousals, 1)
        _, pred_valances = torch.max(proba_valances, 1)
        acc_arousal = (pred_arousals == y_arousals).float().mean()
        acc_valance = (pred_valances == y_valances).float().mean()

        proba_arousals = proba_arousals.numpy()
        proba_valances = proba_valances.numpy()

        y_arousals = y_arousals.numpy()
        y_valances = y_valances.numpy()

        arousal_class_num = 3
        valance_class_num = 3
        # arousal_class_num = 5
        # valance_class_num = 5
        y_arousals_one_hot = label_binarize(y_arousals, np.arange(arousal_class_num))
        y_valances_one_hot = label_binarize(y_valances, np.arange(valance_class_num+1))[:,:-1]

        fpr, tpr, thresholds = metrics.roc_curve(y_arousals_one_hot.ravel(), proba_arousals.ravel())
        auc_arousal = metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(y_valances_one_hot.ravel(), proba_valances.ravel())
        auc_valance = metrics.auc(fpr, tpr)

        logger.debug(f'{subset}, '
                     f'acc_arousal {acc_arousal:.4f}, acc_valance {acc_valance:.4f}, '
                     f'auc_arousal {auc_arousal:.4f}, auc_valance {auc_valance:.4f}')
    
        # mpl.rcParams['font.sans-serif'] = u'SimHei'
        # mpl.rcParams['axes.unicode_minus'] = False
    
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
        # plt.title(u'valance ROC and AUC', fontsize=17)
        # plt.show()