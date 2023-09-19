
model_name_or_path=/ssd1/ywchen/LLM/chatglm2/chatglm2-6b/
template=chatglm2
lora_target=dense,dense_4h_to_h,dense_h_to_4h,query_key_value
device=7

pretrain_dir=output/ihin/pt_ihin/
sft_output_dir=output/ihin/sft_ihin/

## Pre-Training 
CUDA_VISIBLE_DEVICES=$device python src/train_bash.py \
    --stage pt \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --do_train \
    --dataset ihin_pt \
    --finetuning_type lora \
    --lora_target  $lora_target \
    --output_dir $pretrain_dir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --num_train_epochs 20.0 \
    --plot_loss \
    --fp16

## Supervised Fine-Tuning
## CUDA_VISIBLE_DEVICES="0,1,2,3,5" accelerate launch --config_file sh/accelerate_config.yaml  src/train_bash.py \
CUDA_VISIBLE_DEVICES=$device python src/train_bash.py \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --dataset ihin_qa_adn,self_cognition,Ihin_sft_jy0814 \
    --checkpoint_dir $pretrain_dir \
    --finetuning_type lora \
    --lora_target $lora_target  \
    --output_dir $sft_output_dir \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --val_size 0.01 \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --do_train \
    --save_total_limit 10 \
    --ddp_find_unused_parameters false \
    --log_on_each_node False


## SFT inference
CUDA_VISIBLE_DEVICES=$device python src/train_bash.py \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --do_predict \
    --dataset self_cognition,ihin_qa_adn,ihin_qa_test10 \
    --checkpoint_dir $sft_output_dir \
    --output_dir $sft_output_dir \
    --overwrite_cache \
    --per_device_eval_batch_size 6 \
    --predict_with_generate

