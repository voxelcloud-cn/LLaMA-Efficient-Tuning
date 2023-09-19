model_name_or_path=/ssd1/ywchen/LLM/chatglm2/chatglm2-6b/
lora_target=dense,dense_4h_to_h,dense_h_to_4h,query_key_value
template=chatglm2
device=7

# pretrain_dir=output/medical/pt_medical/
sft_output_dir=output/medical/sft_medical/
rm_output_dir=output/medical/rm_medical/
ppo_output_dit=output/medical/ppo_medical/

## Supervised Fine-Tuning
## --dataset medical_finetune,HuatuoGPT_sft,belle_multiturn,alpaca_zh
CUDA_VISIBLE_DEVICES="0,1,2,3,7" accelerate launch --config_file sh/accelerate_config.yaml  src/train_bash.py \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --dataset HuatuoGPT_sft \
    --finetuning_type lora \
    --lora_target  $lora_target \
    --output_dir $sft_output_dir \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.01 \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --do_train \
    --save_total_limit 10 \
    --ddp_find_unused_parameters false \
    --log_on_each_node False

## Reward Modeling
CUDA_VISIBLE_DEVICES=$device python src/train_bash.py \
    --stage rm \
    --model_name_or_path $model_name_or_path \
    --resume_lora_training False \
    --checkpoint_dir  $sft_output_dir \
    --template $template \
    --dataset medical_reward_train \
    --finetuning_type lora \
    --lora_target  $lora_target \
    --output_dir $rm_output_dir \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.01 \
    --plot_loss \
    --fp16 \
    --do_train \
    --save_total_limit 10 \
    --ddp_find_unused_parameters false \
    --log_on_each_node False

## PPO Training
CUDA_VISIBLE_DEVICES="0,1,2,3,7" accelerate launch --config_file sh/accelerate_config.yaml src/train_bash.py \
    --stage ppo \
    --dataset HuatuoGPT_sft \
    --max_samples  40000 \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --resume_lora_training False \
    --checkpoint_dir $sft_output_dir \
    --reward_model $rm_output_dir \
    --finetuning_type lora \
    --lora_target  $lora_target \
    --output_dir $ppo_output_dit \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.5 \
    --max_source_length 256 \
    --max_target_length 128 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.01 \
    --plot_loss \
    --do_train \
    --save_total_limit 10 \
    --ddp_find_unused_parameters false \
    --log_on_each_node False  