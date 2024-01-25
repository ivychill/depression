#!/usr/bin/env bash

case $1 in
    prepare)
#        python prepare_dataset.py
        python resample.py --sr 32000 --n_splits 20
        python trans.py
	      ;;

    train)
#        CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ResNet50.yml
        CUDA_VISIBLE_DEVICES=0 python train_2l.py --config configs/ResNet50.yml
        ;;

    metric)
        CUDA_VISIBLE_DEVICES=0 python metric.py --config configs/ResNet50.yml
        ;;

    test_pretext)
        CUDA_VISIBLE_DEVICES=0 python test_pretext.py --config configs/ResNet50.yml
        ;;

    metric_pretext)
        CUDA_VISIBLE_DEVICES=0 python metric_pretext.py --config configs/ResNet50.yml
        ;;

    *)
        echo "wrong argument"
		    exit 1
        ;;
esac
exit 0
