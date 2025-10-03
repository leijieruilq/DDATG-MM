export CUDA_VISIBLE_DEVICES=$3

all_models=("PatchTST")
start_index=$1
end_index=$2
models=("${all_models[@]:$start_index:$end_index-$start_index+1}")
root_paths=("./data/Algriculture")
data_paths=("US_RetailBroilerComposite_Month.csv") 
pred_lengths=(6 8 10 12)
seeds=(2021)
alphas=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
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
        for alpha in "${alphas[@]}"  # 新增alpha循环
        do
          root_path=${root_paths[$i]}
          data_path=${data_paths[$i]}
          model_id=$(basename ${root_path})

          echo "Running model $model_name with root $root_path, data $data_path, pred_len $pred_len, alpha $alpha"
          python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $root_path \
            --data_path $data_path \
            --model_id ${model_id}_gnn_${seed}_24_${pred_len}_fullLLM_${use_fullmodel}_alpha${alpha} \
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
            --save_name "result_algriculture_bert_gnn" \
            --llm_model BERT \
            --use_gnn 1 \
            --alpha $alpha \
            --huggingface_token 'NA' \
            --use_fullmodel $use_fullmodel
        done
      done
    done
  done
done