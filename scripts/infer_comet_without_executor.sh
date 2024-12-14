if [ "$#" -eq  "0" ]
then
    echo "No arguments supplied"
else
    CUDA_VISIBLE_DEVICES=$1 python infer.py \
        --config_name comet_without_executor \
        --inputter_name comet_without_executor \
        --add_nlg_eval \
        --seed 13 \
        --load_checkpoint data/comet_without_executor.comet_without_executor/2023-02-11100701.2e-05.20.1gpu/epoch-2.bin \
        --fp16 false \
        --max_input_length 512 \
        --max_decoder_input_length 40 \
        --max_length 512 \
        --min_length 1 \
        --infer_batch_size 20 \
        --infer_input_file ./_reformat/test.txt \
        --temperature 0.7 \
        --top_k 30 \
        --top_p 0.3 \
        --num_beams 1 \
        --repetition_penalty 1.03 \
        --no_repeat_ngram_size 2
fi