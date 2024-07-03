SEEDS_FILE="../seeds.txt"





# Read each line from the seeds file
i=1
while IFS= read -r seed
do
    echo "Running experiment $i with seed $seed"
    CUDA_VISIBLE_DEVICES=3,4 python train.py \
        -ptm "MCG-NJU/videomae-base-finetuned-kinetics" \
        --gpus 2 \
        -sched "plateau" \
        -lr 0.0001 \
        --data_path "../slovo_tensors_32" \
        --exp_name "slovo_pretrain_seed_$seed" \
        --frames 16 \
        --random_sample \
        --dataset "slovo" \
        -t "normalize" \
        --seed "$seed"
    
    ((i++))
    # Exit loop after 10 runs
    if [ $i -gt 10 ]; then
        break
    fi
done < "$SEEDS_FILE"
