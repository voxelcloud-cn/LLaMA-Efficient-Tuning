## 脚本说明

原来增量预训练阶段在[MedicalGPT](https://github.com/shibing624/MedicalGPT)中训练  
后续的其他阶段在[ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)中训练  

现在将之前的bash脚本统一调整到[LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)。  
脚本已经过测试，能正常运行。性能是否能与迁移前保持一致，未进行测试。

ihin_run.sh： ihin模型的训练  
medical_run.sh: 通用医疗模型的训练

脚本输出地址请见服务器：/mnt/eye_team/ywchen/model_train/gpt/LLaMA-Efficient-Tuning/output/
相关数据集并未传到github，请在本地目录下查看。

## 新增数据集
ihin_pt：把公众号的markdown 文件合并为一个text文件  
Ihin_sft_jy0814： 这是敬玉用gpt生成的QA数据  
ihin_qa_adn: 这是勇维用文章的小标题和正文生成的QA数据。**只包含部分文章**。  另外提取了《问答系列》的QA。  
cMedQA2_train：见[github](https://github.com/zhangsheng93/cMedQA2)  
medical_reward_train: 来自shibing624/medical，只是修改了数据格式。
