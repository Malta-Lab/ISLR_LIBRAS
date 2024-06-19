CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_normalize" -t "normalize" --frames 16 --random_sample -bs 8

CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_color_jitter" -t "color_jitter" --frames 16 --random_sample -bs 8

CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_random_perspective" -t "random_perspective" --frames 16 --random_sample -bs 8

CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_gaussian_blur" -t "gaussian_blur" --frames 16 --random_sample -bs 8

CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_random_horizontal_flip" -t "random_horizontal_flip" --frames 16 --random_sample -bs 8

CUDA_VISIBLE_DEVICES=3,4 python train.py -ptm "facebook/timesformer-base-finetuned-k400" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "transforms_ablation_aug_mix" -t "aug_mix" --frames 16 --random_sample -bs 8
