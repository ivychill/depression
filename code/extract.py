
import warnings
import src.configuration as C
import src.models as models
from src.utils import get_parser, load_config, get_logger, set_logger, set_seed
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io


if __name__ == "__main__":
    data_source = 'baidu'
    # data_source = 'IEMOCAP'
    # target_name = 'arousal'
    target_name = 'valance'
    
    warnings.filterwarnings("ignore")
    torch.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=np.inf)

    args = get_parser().parse_args()
    config = load_config(args.config)
    global_params = config["globals"]
    output_dir = Path(global_params["output_dir"])

    logger = get_logger()
    set_logger(logger, output_dir / "extract.log")
    set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])
    model = models.get_model(config).to(device)

    fold_num = 5
    for index in range(fold_num):
        base_dir = output_dir / f'fold{index}'
        # logger.info(f'random weights')
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

        for subset in ['train', 'val']:
            path = base_dir/f'{subset}_{target_name}.csv'
            logger.info(f'path {path}')
            meta_data = pd.read_csv(path)
            data_dir = Path(config["data"]["train_audio_path"])
            loader = C.get_loader(meta_data, data_dir, config, "valid")

            features = torch.FloatTensor()
            arousals = torch.FloatTensor()
            valances = torch.FloatTensor()

            for data in tqdm(loader):
                images = data['image'].to(device)
                labels = data['targets'].to(device)
                with torch.no_grad():
                    output = model(images)
                feature = output['feature']
                # feature = feature[::2]
                arousal = labels[:, 0].float()
                valance = labels[:, 1].float()
                features = torch.cat((features, feature.detach().cpu()), 0)
                arousals = torch.cat((arousals, arousal.detach().cpu()), 0)
                valances = torch.cat((valances, valance.detach().cpu()), 0)

            feat_path = base_dir/f'{subset}_{target_name}.mat'
            result = {'feature': features.numpy(),
                      'arousal': arousals.numpy(),
                      'valance': valances.numpy()}
            scipy.io.savemat(feat_path, result)