from sklearn.cluster import DBSCAN
import numpy as np
from pathlib import Path
from numpy import linalg as LA
import pandas as pd


feature_file = Path('../user_data/baidu/model_data/feature.npy')
features = np.load(feature_file)
print(f'features {features.shape}')
norm = LA.norm(features, axis=1)
# print(f'norm {norm}')
clustering = DBSCAN(eps=0.56, min_samples=3).fit(features)
labels = clustering.labels_

# print(f'labels {labels}')

unique, counts = np.unique(labels, return_counts=True)
cluster_dict = dict(zip(unique, counts))
print(f'cluster {len(cluster_dict)}, {cluster_dict}')

index_0 = np.where(labels == 0)[0]
print(f'index_0 {index_0}')
val_path = Path('../user_data/baidu/train_3_2_plus.csv')
df = pd.read_csv(val_path)
print(df.iloc[index_0, 0])