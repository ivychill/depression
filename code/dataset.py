# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as data
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from log import get_logger


logger = get_logger()

class NpyDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path):
        psy_dict = {'暂无问题': 0, '心理困扰': 1, '心理危机': 1}    # todo
        gender_dict = {'男': 0, '女': 1}
        self.data = []
        data_root = datadir.absolute()
        # dates = os.listdir(data_root)
        date_paths = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
        for date_path in sorted(date_paths):
            # date_path = data_root/date
            _, date = os.path.split(date_path)
            audios = os.listdir(date_path)
            for audio in sorted(audios):
                audio_path = date_path/audio
                id = audio[:audio.find('-')]
                index = np.where((df['日期'].replace('\.', '', regex=True) == date) & (df['编号'] == int(id)))   # index as id_class
                if len(index[0]) == 1:
                    psy_eval = df.iloc[index[0][0]]['危机评估']    # -3 refer to 危机评估
                    psy = psy_dict[psy_eval]
                    gender_value = df.iloc[index[0][0]]['性别']
                    gender = gender_dict[gender_value]
                    age = df.iloc[index[0][0]]['年龄']
                    # logger.debug(f'appending {audio_path}, {index[0][0]}, {psy_class}')
                    self.data.append((audio_path, index[0][0], gender, age, psy))  # index[0][0] refer to id

        self.height = 224
        self.width = 448
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width), Image.BILINEAR),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image = np.load(self.data[idx][0])  # (3, 224, 547)
        image = self.transformer(torch.from_numpy(image))
        id = self.data[idx][1]
        gender = self.data[idx][2]
        age = self.data[idx][3]
        psy =  self.data[idx][4]
        return image, (id, gender, age, psy)