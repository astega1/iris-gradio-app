Data used for finetuning: 

Finetome100K 
& 
flytech/python-codes-25k 

We almost used every data point in the flytech dataset. The notebook for the first iteration was configured a little bit different since we downloaded the peft module to set lora params. We also trained on a quantsiezed version. 

We used Unsloth and orginal LLM was LLama-3B instruct. 

We did it in Kaggle for GPU support, we saved checkpoints in Huggingface in order to continue fintuning. 

The gradio app we devloped can be found using the following link: https://huggingface.co/spaces/astegaras/iris 
