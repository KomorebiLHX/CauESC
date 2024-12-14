if [ "$#" -eq "0" ]; then
    echo "No arguments supplied"
else
    CUDA_VISIBLE_DEVICES=$1 python train_multi.py \
        --config_name causal \
        --inputter_name causal \
        --eval_input_file ./_reformat/valid.txt \
        --seed 42 \
        --max_input_length 512 \
        --max_decoder_input_length 512 \
        --train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --eval_batch_size 16 \
        --learning_rate 3e-5 \
        --num_epochs 5 \
        --warmup_steps 120 \
        --fp16 false \
        --loss_scale 0.0 \
        --pbar true
fi
