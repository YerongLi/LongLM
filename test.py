import llama_self_extend_patch as LlamaSE
from modify_utils import modify_method_of_instance
from functools import partial

# Load your model, e.g., loaded_model = AutoModelForCausalLM.from_pretrained(model_path) 

# group_size_1 is group_window, group_size_2 is neighbor_window
self_extend_forward = partial(LlamaSE.self_extend_forward, group_size_1=4, group_size_2=1024)
modify_method_of_instance(loaded_model, "LlamaAttention", "forward", self_extend_forward)

# Inference, e.g., loaded_model.generate(...)

