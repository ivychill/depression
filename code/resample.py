import argparse
import soundfile as sf
import warnings
import librosa
import pandas as pd
import os
import numpy as np
from pathlib import Path
from joblib import delayed, Parallel


def resample(df: pd.DataFrame, target_sr: int):
    audio_dir = Path("../user_data/IEMOCAP/train")
    resample_dir = Path("../user_data/IEMOCAP/train_resampled")
    resample_dir.mkdir(exist_ok=True, parents=True)
    warnings.simplefilter("ignore")

    audio_paths = os.listdir(audio_dir)
    for audio_path in audio_paths:
        audio_name = os.path.basename(audio_path)
        index = np.where(df.loc[:, ['audio']] == audio_name)
        if len(index[0]) > 0:
            print(audio_name)
            try:
                y, _ = librosa.load(
                    audio_dir / audio_name,
                    sr=target_sr, mono=True, res_type="kaiser_fast")
                sf.write(resample_dir / audio_name, y, samplerate=target_sr)
            except Exception:
                with open(resample_dir/"skipped.txt", "a") as f:
                    file_path = str(audio_dir / audio_name)
                    f.write(file_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", default=22050, type=int)
    parser.add_argument("--n_splits", default=12, type=int)
    args = parser.parse_args()

    target_sr = args.sr

    train = pd.read_csv("../user_data/IEMOCAP/clean_5.csv")
    dfs = []
    for i in range(args.n_splits):
        if i == args.n_splits - 1:
            start = i * (len(train) // args.n_splits)
            df = train.iloc[start:, :].reset_index(drop=True)
            dfs.append(df)
        else:
            start = i * (len(train) // args.n_splits)
            end = (i + 1) * (len(train) // args.n_splits)
            df = train.iloc[start:end, :].reset_index(drop=True)
            dfs.append(df)

    Parallel(
        n_jobs=args.n_splits,
        verbose=10)(delayed(resample)(df, args.sr) for df in dfs)

    print('audio has been resampled!')