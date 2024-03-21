# transfromers version 4.32.0
import warnings
warnings.filterwarnings("ignore")

import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial
import json
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
original_llama_forward = LlamaAttention.forward
self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=8, group_size_2=1024)


# model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# model_path = '/scratch/yerong/.cache/pyllama/TinyLlama-1.1B-Chat-v1.0'
model_path = os.getenv('TINY')
# model_path = os.getenv('CHATLM')
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
print(model.device)
print('device =============')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


for line in open("passkey_examples_5k.jsonl", "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print( "-----------------------------------" )
    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
    print( "Passkey target:", example["target"] )


    modify_method_of_instance(model, "LlamaAttention", "forward", original_llama_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    # print(input_ids)
    # tensor([[    1,  1670,   338,  ...,  1820,   338, 29871]], device='cuda:0')
    answer= "Llama2:     [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
    answer = answer.replace("\n", "\\n")
    print( answer )


    modify_method_of_instance(model, "LlamaAttention", "forward", self_extend_forward)
    tokens = model.generate(input_ids, max_new_tokens=6)
    answer= "SelfExtend: [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
    answer = answer.replace("\n", "\\n")
    print( answer )
    print( "-----------------------------------\n" )






