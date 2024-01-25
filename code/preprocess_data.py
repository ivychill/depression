import sys
import warnings
from pathlib import Path
import os
import shutil
import numpy as np
import librosa
import cv2
from log import get_logger, set_logger



def reorganize_audio_path():
    def clear(dst):
        exts = [".mp3", ".m4a", '.aac', '.wav']
        dst_paths = [p for p in sorted(dst.iterdir()) if p.suffix in exts]
        for dst_path in dst_paths:
            os.remove(dst_path)
    
    def append(src, dst):
        exts = [".mp3", ".m4a", '.aac', '.wav']
        src_paths = [p for p in sorted(src.iterdir()) if p.suffix in exts]
        dst_paths = [p for p in sorted(dst.iterdir()) if p.suffix in exts]
        last_sn = 0

        sns = []
        for dst_path in dst_paths:
            _, file = os.path.split(dst_path)
            index = file.find('-')
            sn = int(file[:index])
            sns.append(sn)
            last_sn = max(sns)
        
        logger.debug(f'last_sn {last_sn}')
        
        for src_path in src_paths:
            _, src_file = os.path.split(src_path)
            src_index = src_file.find('-')
            src_sn = int(src_file[:src_index])
            dst_sn = last_sn + src_sn
            dst_file = str(dst_sn) + src_file[src_index:]
            dst_path = dst/dst_file
            logger.debug(f'copying {src_path} to {dst_path}')
            shutil.copy(src_path, dst_path)
    
    def rm_file_prefix(src):
        exts = [".mp3", ".m4a", '.aac', '.wav']
        src_paths = [p for p in sorted(src.iterdir()) if p.suffix in exts]
        for src_path in src_paths:
            src_dir, src_file = os.path.split(src_path)
            if src_file.count('-') > 1:
                src_index = src_file.find('-')
                dst_file = src_file[(src_index+1):]
                dst_path = Path(src_dir)/dst_file
                logger.debug(f'moving {src_path} to {dst_path}')
                os.rename(src_path, dst_path)
        
    data_dir = Path("../data")
    
    date_dir = data_dir/'20231104'
    clear(date_dir)
    append(date_dir/'第1组', date_dir)
    append(date_dir/'第2组', date_dir)
    # rm_file_prefix(date_dir/'第3组')
    append(date_dir/'第3组', date_dir)
    
    date_dir = data_dir/'20231105'
    clear(date_dir)
    append(date_dir/'第一组', date_dir)
    append(date_dir/'第二组', date_dir)        
    append(date_dir/'第三组', date_dir)
    
    date_dir = data_dir/'20231112'
    clear(date_dir)
    append(date_dir/'鼓楼校区', date_dir)
    append(date_dir/'仙林校区', date_dir)        
    
    date_dir = data_dir/'20231118'
    clear(date_dir)
    append(date_dir/'浦口校区', date_dir)
    append(date_dir/'仙林校区', date_dir)        


def slice_audio():
    def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
        # Stack X as [X,X,X]
        X = np.stack([X, X, X], axis=-1)

        # Standardize
        mean = mean or X.mean()
        X = X - mean
        std = std or X.std()
        Xstd = X / (std + eps)
        _min, _max = Xstd.min(), Xstd.max()
        norm_max = norm_max or _max
        norm_min = norm_min or _min
        if (_max - _min) > eps:
            # Normalize to [0, 255]
            V = Xstd
            V[V < norm_min] = norm_min
            V[V > norm_max] = norm_max
            V = 255 * (V - norm_min) / (norm_max - norm_min)
            V = V.astype(np.uint8)
        else:
            # Just zero
            V = np.zeros_like(Xstd, dtype=np.uint8)
        return V
    

    def transform_voice(fname):
        melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 20,
        "fmax": 16000
        }
        
        SR = 48000
        slice_duration = 4
        img_size = 224
        y, SR = librosa.load(fname,
                            sr=SR,
                            mono=True,
                            res_type="kaiser_fast")
    
        y = y.astype(np.float32)
        len_y = len(y)
        # omit beginning 8s and ending 4s
        start = int(SR * 8)
        # start = int(SR * slice_duration * 2)
        end = start + SR * slice_duration
        images = []
        # while len_y > start:
        while len_y > end + int(SR * 4):
        # while len_y > end + int(SR * slice_duration):
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (SR * slice_duration):
                break
            melspec = librosa.feature.melspectrogram(y=y_batch, sr=SR, **melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)
            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * img_size / height), img_size))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
            images.append(image)
            # print(f'start {start/SR}, end {end/SR}')
            # start = start + int(SR * slice_duration)
            # end = end + int(SR * slice_duration)
            start = start + int(SR * slice_duration / 4)
            end = end + int(SR * slice_duration / 4)

        return images

    src_root = Path('../data/')
    dst_root = Path('../user_data/')
    os.makedirs(dst_root, exist_ok=True)

    src_paths = [p for p in sorted(src_root.iterdir()) if p.is_dir()]
    for src_path in src_paths:
        _, src_date = os.path.split(src_path)
        dst_path = dst_root/src_date
        os.makedirs(dst_path, exist_ok=True)
        exts = [".mp3", ".m4a", '.aac', '.wav']
        src_audios = [p for p in sorted(src_path.iterdir()) if p.suffix in exts]
        for src_audio in src_audios:
            _, file = os.path.split(src_audio)
            prefix = file
            # prefix, _ = os.path.splitext(file)
            logger.debug(f'slicing {src_audio}')
            images = transform_voice(src_audio)
            for i, image in enumerate(images):
                dst_npy = dst_path/f'{prefix}_{i}.npy'
                # image shape (3, 224, 658)
                # logger.debug('image', image.shape, image[0, 0:5, 0:5])   # (3, 224, 658)
                np.save(dst_npy, image)

    logger.debug('melspectrogram end')


def format_table():
    pass


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    logger = get_logger()
    log_dir = Path('../log')
    log_dir.mkdir(parents=True, exist_ok=True)
    set_logger(logger, log_dir / "preprocess.log")
    # reorganize_audio_path()
    slice_audio()