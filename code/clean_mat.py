from pathlib import Path
import os
import hdf5storage
import numpy as np
from shutil import copyfile


base_dir = Path('../user_data/baidu')
# base_dir = Path('../user_data/IEMOCAP')
source_dir = base_dir/'train_mat_800_slide'
target_dir = base_dir/'clean_mat_800_slide'
target_dir.mkdir(exist_ok=True, parents=True)

count = 0
audios = os.listdir(source_dir)
for audio in sorted(audios):
    mat = hdf5storage.loadmat(str(source_dir/audio))
    if np.isnan(mat['data']).any():
        print(audio)
        count += 1
    else:
        copyfile(source_dir/audio, target_dir/audio)

print(f'count {count}')