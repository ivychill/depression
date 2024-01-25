import logging
import time
import warnings
import src.configuration as C
import src.utils as utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from fastprogress import progress_bar
from torchvision import models


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


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x).view(batch_size, -1)
        cls_list = list(self.classifier.children())
        out = cls_list[0](feature)
        out = cls_list[1](out)
        logits = self.classifier(feature)
        multiclass_proba = F.softmax(logits, dim=1)
        multilabel_proba = F.sigmoid(logits)
        return {
            "feature": out,
            "logits": logits,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


def get_model(config: dict, weights_path: str):
    model = ResNet(**config)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = utils.get_logger()
    utils.set_logger(logger, output_dir / "feat.log")
    utils.set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])
    # model = models.get_model(config).to(device)

    N_CLASS = 52
    model_config = {
        "base_model_name": "resnet50",
        "pretrained": False,
        "num_classes": N_CLASS
    }

    weights_path = "../../zj/output/b_ResNet50/fold0/checkpoints/train.93.pth"
    logger.info(f'weights_path {weights_path}')
    model = get_model(model_config, weights_path)

    val_path = Path('../user_data/baidu/train_3_2_plus.csv')
    logger.info(f'val_path {val_path}')
    meta_data = pd.read_csv(val_path)
    data_dir = Path(config["data"]["train_audio_path"])
    loader = C.get_loader(meta_data, data_dir, config, "valid")

    features = torch.FloatTensor()
    for data in progress_bar(loader):
        images = data.to(device)
        output = model(images)
        feature = output["feature"]
        features = torch.cat((features, feature.detach().cpu()), 0)

    logger.debug(f'features {features.size()}')
    features = F.normalize(features, p=2, dim=1).numpy()
    np.save(str(output_dir/'feature.npy'), features)
