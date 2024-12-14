if [ "$#" -eq "0" ]; then
    echo "No arguments supplied"
else
    CUDA_VISIBLE_DEVICES=$1 python train_multi.py \
        --config_name comet_without_other_effect \
        --inputter_name comet_without_other_effect \
        --eval_input_file ./_reformat/valid.txt \
        --seed 13 \
        --max_input_length 512 \
        --max_decoder_input_length 40 \
        --train_batch_size 20 \
        --gradient_accumulation_steps 1 \
        --eval_batch_size 50 \
        --learning_rate 2e-5 \
        --num_epochs 8 \
        --warmup_steps 120 \
        --fp16 false \
        --loss_scale 0.0 \
        --pbar true
fi
