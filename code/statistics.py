import pandas as pd
import math
import numpy as np


def count_clean(file_name):
    meta_data = pd.read_csv(file_name)

    arousal = meta_data['arousal'].to_numpy()
    valance = meta_data['valance'].to_numpy()

    count = 0
    for index in range(len(arousal)):
        arousal_decimal, arousal_integer = math.modf(arousal[index])
        valance_decimal, valance_integer = math.modf(valance[index])
        print(f'index {index}, {arousal_decimal}, {math.isclose(arousal_decimal, 0, abs_tol=0.1)}')
        print(f'index {index}, {valance_decimal}, {math.isclose(valance_decimal, 0, abs_tol=0.1)}')

        if math.isclose(arousal_decimal, 0, abs_tol=0.1) and math.isclose(valance_decimal, 0, abs_tol=0.1):
            count += 1

    print(f'count {count}')
    return count


def count_unique(file_name):
    wav_files = []
    for line in open(file_name):
        wav_file = line[:(line.index(line.split('_')[-1])-1)]
        # print(f'wav_file {wav_file}')
        wav_files.append(wav_file)
    wav_file_set = set(wav_files)
    print(f'file_name {file_name}, list {len(wav_files)}, set {len(wav_file_set)}')
    # with open('../user_data/IEMOCAP/set.txt', 'w') as f:
    #     for item in sorted(list(wav_file_set)):
    #         f.write("%s\n" % item)

def count_label(file_name):
    meta_data = pd.read_csv(file_name)

    arousal = meta_data['arousal'].to_numpy()
    valance = meta_data['valance'].to_numpy()
    unique, counts = np.unique(arousal, return_counts=True)
    print(f'arousal {dict(zip(unique, counts))}')
    unique, counts = np.unique(valance, return_counts=True)
    print(f'valance {dict(zip(unique, counts))}')


if __name__ == "__main__":
    # count_clean('../user_data/IEMOCAP/train_5.csv')
    # count_unique('../user_data/IEMOCAP/train_npy_1s.txt')
    # count_unique('../user_data/IEMOCAP/train_npy_3s.txt')
    # count_unique('../user_data/IEMOCAP/train_npy_5s.txt')
    count_label('../user_data/IEMOCAP/clean_5.csv')

#
# print(type())
# for index in range(5):
#     count = meta_data[meta_data["arousal"] == index].count()
#     print(index, count)