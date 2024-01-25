prepare:
	python prepare_dataset.py
	python resample.py --sr 32000 --n_splits 20
	python trans.py

train:
	CUDA_VISIBLE_DEVICES=1 python train.py --config configs/ResNet50.yml

train_2l:
	CUDA_VISIBLE_DEVICES=3 python train_2l.py --config configs/ResNet50.yml

metric:
	CUDA_VISIBLE_DEVICES=0 python metric.py --config configs/ResNet50.yml

extract:
	CUDA_VISIBLE_DEVICES=0 python extract.py --config configs/ResNet50.yml

test:
	CUDA_VISIBLE_DEVICES=0 python test_pretext.py --config configs/ResNet50.yml

metric_pretext:
	CUDA_VISIBLE_DEVICES=0 python metric_pretext.py --config configs/ResNet50.yml

feat:
	CUDA_VISIBLE_DEVICES=0 python gen_feature.py --config configs/ResNet50.yml