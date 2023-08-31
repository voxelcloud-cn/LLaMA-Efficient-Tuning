MASTER_PORT=$(shuf -n 1 -i 10000-65535)

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 --master_port $MASTER_PORT src/train_bash.py \
    --deepspeed deepspeed_zero2.json \
    --stage ppo \
    --model_name_or_path /mnt/eye_team/jyhu/LLaMA-Efficient-Tuning/models/chatglm2-6b \
    --do_train \
    --dataset self_cognition,Ihin_sft,ihin_qa_adn \
    --dataset_dir /mnt/eye_team/jyhu/Ihin_assets \
    --template chatglm2 \
    --finetuning_type lora \
    --lora_target query_key_value \
    --resume_lora_training False \
    --checkpoint_dir ../outputs/Ihin/Ihin-sft-chatglm2/checkpoint-24000 \
    --reward_model ../outputs/Ihin/Ihin-rm-chatglm2/checkpoint-800 \
    --output_dir ../outputs/Ihin/Ihin-ppo-chatglm2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --fp16 \
    --plot_loss \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --val_size 0.01 \
    --preprocessing_num_workers 32 \
    # --max_samples 2000 \