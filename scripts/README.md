# run continual pretraining for chatglm2-6b with deepspeed zero2
```bash 
cd scripts
bash run_Ihin_pt.sh
```
# run supervised finetuning for chatglm2-6b with deepspeed zero2
```bash 
cd scripts
bash run_Ihin_sft.sh
```
# train reward model 
```bash 
cd scripts
bash run_rm.sh
```

# run proximal policy optimization for chatglm2-6b with deepspeed zero2
```bash 
cd scripts
bash run_ppo.sh
```

# run continual pretraining for llama2-70b with deepspeed zero3
```bash 
cd scripts
bash run_llama2-70b_pt.sh
```