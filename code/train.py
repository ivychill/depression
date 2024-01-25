
import warnings
import src.callbacks as clb
import src.configuration as C
import src.models as models
from src.utils import get_parser, load_config, get_logger, set_logger, set_seed
# import time
from catalyst.dl import SupervisedRunner
from pathlib import Path
import torch
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # target_name = 'arousal'
    # target_name = 'valance'
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

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        (output_dir/f"fold{i}").mkdir(exist_ok=True, parents=True)
        trn_df.to_csv(output_dir / f"fold{i}"/f'train_{target_name}.csv', index=None)
        val_df.to_csv(output_dir / f"fold{i}"/f'val_{target_name}.csv', index=None)

        # if i not in global_params["folds"]:
        #     continue

        loaders = {
            phase: C.get_loader(df_, datadir, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }

        # loaders = {
        #     "train": C.get_loader(df, datadir, config, "train")
        # }

        model = models.get_model(config).to(device)
        # model.float()
        criterion = C.get_criterion(config).to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)
        callbacks = clb.get_callbacks(config)

        runner = SupervisedRunner(
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])

        runner.train(
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=output_dir / f"fold{i}",
            # logdir=output_dir,
            # default for contrast
            callbacks=callbacks,
            main_metric=global_params["main_metric"],
            minimize_metric=global_params["minimize_metric"],
        )
