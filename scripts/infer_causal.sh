if [ "$#" -eq "0" ]; then
    echo "No arguments supplied"
else
    CUDA_VISIBLE_DEVICES=$1 python infer.py \
        --config_name causal \
        --inputter_name causal \
        --add_nlg_eval \
        --seed 42 \
        --load_checkpoint data/causal.causal/2023-01-28073925.3e-05.16.1gpu/epoch-2.bin \
        --fp16 false \
        --max_input_length 512 \
        --max_decoder_input_length 512 \
        --max_length 512 \
        --min_length 5 \
        --infer_batch_size 32 \
        --infer_input_file ./_reformat/test.txt \
        --temperature 0.7 \
        --top_k 0 \
        --top_p 0.9 \
        --num_beams 1 \
        --repetition_penalty 1.0 \
        --no_repeat_ngram_size 0
fi
