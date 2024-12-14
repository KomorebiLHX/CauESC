if [ "$#" -eq  "0" ]
then
    echo "No arguments supplied"
else
    CUDA_VISIBLE_DEVICES=$1 python infer.py \
        --config_name comet \
        --inputter_name comet \
        --add_nlg_eval \
        --seed 42 \
        --load_checkpoint data/comet.comet/2023-02-01055048.2e-05.20.1gpu/epoch-7.bin \
        --fp16 false \
        --max_input_length 512 \
        --max_decoder_input_length 40 \
        --max_length 512 \
        --min_length 1 \
        --infer_batch_size 1 \
        --infer_input_file ./_reformat/test.txt \
        --temperature 0.7 \
        --top_k 30 \
        --top_p 0.3 \
        --num_beams 1 \
        --repetition_penalty 1.03 \
        --no_repeat_ngram_size 2
fi