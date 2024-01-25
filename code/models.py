import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, dim_id, dim_psy=3, pretrained=False):
        super().__init__()

        base_model = models.__getattribute__(base_model_name)(pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        dim_feature = 256

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(256, dim_feature), nn.ReLU(), nn.Dropout(p=0.2))

        self.out_id = nn.Linear(dim_feature, dim_id)
        self.out_gender = nn.Linear(dim_feature, 2)
        self.out_age = nn.Linear(dim_feature, 1)
        self.out_psy = nn.Linear(dim_feature, dim_psy)

    def forward(self, x):
        # 1s: [3, 224, 110]; 3s: [3, 224, 329]; 5s: [3, 224, 547]
        if len(x.size()) > 4:  # contrast
            x = torch.reshape(x, (-1, x.size()[2], x.size()[3], x.size()[4]))
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        feature = self.classifier(x)
        logits_id = self.out_id(feature)
        logits_gender = self.out_gender(feature)
        age_pred = self.out_age(feature)
        logits_psy = self.out_psy(feature)

        return logits_id, logits_gender, age_pred, logits_psy