CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "1_sample_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 1

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "5_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 5

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "10_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 10

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "15_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 15

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "20_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 20

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "25_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 25

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "30_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 30

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "35_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 35

CUDA_VISIBLE_DEVICES=1,2 python train.py -ptm "MCG-NJU/videomae-base-finetuned-kinetics" --gpus 2 -sched "plateau" -lr 0.0001 --data_path "../MINDS_tensors_32" --exp_name "40_samples_per_class" -t "color_jitter" "random_perspective"  "random_horizontal_flip" "gaussian_blur" "normalize" --frames 16 --random_sample -bs 16 --n_samples_per_class 40
