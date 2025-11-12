from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import openai
import yaml
import time

from typing import Union, List, Dict


def load_gpt_model(openai_config_path, model_name, api_key_index=0):
    with open(openai_config_path, 'r') as file: config = yaml.safe_load(file)['default']
    usable_keys = []
    for item in config:
        if item.get('azure_deployment', model_name) == model_name:
            if 'azure_deployment' in item: del item['azure_deployment']
            #print('Found usable key', len(usable_keys), ':', item)
            usable_keys.append(item)
    #print('\nUsing key', api_key_index, ':', usable_keys[api_key_index])
    client_class = usable_keys[api_key_index]['client_class']
    del usable_keys[api_key_index]['client_class']
    return eval(client_class)(**usable_keys[api_key_index])

def get_openai_completion_with_retry(client, sleepsec=10, **kwargs) -> str:
    while 1:
        try: return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            print('OpenAI API error:', e, 'sleeping for', sleepsec, 'seconds', flush=True) 
            time.sleep(sleepsec)
            if "400" in str(e):
                return "OpenAI Rejected"
            

class Model():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        if "azure" not in model_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                use_fast=True, 
                trust_remote_code=True, 
                token=os.getenv("HF_TOKEN"), 
                cache_dir=os.getenv("HF_HOME")
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                device_map="auto", 
                # output_attentions=output_attentions,
                # attn_implementation="flash_attention_2",
                # attn_implementation="sdpa",
                dtype="auto", 
                token=os.getenv("HF_TOKEN"), 
                cache_dir=os.getenv("HF_HOME")
            )
            self.model.eval()

            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token     
        else:
            model_name = model_name_or_path.split("/")[-1]
            self.model_name = model_name
            openai_config_path = f"configs/azure_configs/{model_name}.yaml"
            api_key_index = 0
            self.model = load_gpt_model(openai_config_path, model_name, api_key_index)
            self.tokenizer = None

    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
    ):
        if isinstance(messages, list):
            for message in messages:
                if "{optim_str}" in message["content"]:
                    message["content"] = message["content"].replace("{optim_str}", "")
        else:
            messages = messages.replace("{optim_str}", "")

        if "azure" not in self.model_name_or_path.lower():

            if isinstance(messages, list):
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
                attention_mask = torch.ones_like(input_ids).to(self.model.device)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            else:
                # already in the right format
                assert isinstance(messages, str), "messages must be a string if apply_chat_template is False"
                inputs = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=1.2
            )
            generated_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        else:
            generated_text = get_openai_completion_with_retry(self.model, 
                messages=messages,
                model=self.model_name, 
            )            
        return generated_text