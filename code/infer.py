import src.configuration as C
import numpy as np
import torch
from pathlib import Path
from fastprogress import progress_bar
import src.models as models
import src.utils as utils


def extract_feature(model, images, labels):
    dim_emotion = 3
    feature = torch.FloatTensor().to(device)
    output = model(images)
    feature_person = output['feature']
    emotions = labels[:, 2].int().squeeze()

    for em_value in range(dim_emotion):
        indices = (emotions == em_value).nonzero().squeeze()
        conditional_features = feature_person[indices]
        feature_average = torch.mean(conditional_features, axis=0, keepdim=True)
        feature = torch.cat((feature, feature_average), 1)
        # print('conditional_features', conditional_features.size(), conditional_features[:, :4])
        # print('feature_average', feature_average.size(), feature_average[:, :4])
        # print('feature', feature.size(), feature[:, :4])

    return feature


if __name__ == "__main__":
    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = utils.get_logger()
    utils.set_logger(logger, output_dir / "test.log")
    utils.set_seed(global_params["seed"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)

    weights_path = "../user_data/256_balance_regularization/checkpoints/train.494.pth"
    logger.info(f'weights_path {weights_path}')

    device = C.get_device(global_params["device"])
    model = models.get_model(config).to(device)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = C.get_loader(df, datadir, config, "test")

    features = torch.FloatTensor()
    for (images, labels) in progress_bar(loader):
        images = images.squeeze().to(device)
        labels = labels.squeeze().to(device)
        feature = extract_feature(model, images, labels)
        # cad in all labels is the same
        feature = torch.cat((feature, labels[:1,-1:].float()), 1)
        print('feature', feature.size())
        features = torch.cat((features, feature.detach().cpu()), 0)

    np.save(output_dir/'feature_and_cad', features.numpy())