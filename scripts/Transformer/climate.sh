export CUDA_VISIBLE_DEVICES=$3

all_models=("Transformer")
start_index=$1
end_index=$2
models=("${all_models[@]:$start_index:$end_index-$start_index+1}")
root_paths=("./data/Climate")
data_paths=("US_precipitation_month.csv") 
pred_lengths=(6 8 10 12)
seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for ((i=0; i<$length; i++))
    do
      for pred_len in "${pred_lengths[@]}"
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})

        echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 8 \
          --label_len 4 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --prompt_weight 0.1 \
          --pool_type "avg" \
          --save_name "result_climate_bert" \
          --llm_model BERT \
          --huggingface_token 'NA'\
          --use_gnn 0 \
          --use_fullmodel $use_fullmodel
      done
    done
  done
done

