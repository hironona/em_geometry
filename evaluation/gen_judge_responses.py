
# %%

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from gen_eval_util_openai import gen_and_eval, judge_responses, get_responses
import gc
import torch
import yaml

# %%
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

question_file = config["question_file"]    # default first plot questions
model_id = config['model_id']
save_dir = config['save_dir']
save_suffix = config['save_suffix']

save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix}.csv")
# medical_save_path = os.path.join(save_dir, f"med-qu_{model_id.split('/')[-1]}{save_suffix}.csv")
overwrite = True   # if True, overwrites any existing file, otherwise it appends to it

use_json_questions = False
use_template_questions = False
n_per_question = 50
new_tokens = 600
temperature = 1
top_p = 1
cleanup_model = True

# This works for LoRA adapters or full models from HF
model, tokenizer = load_model(model_id)


# GENERATE AND EVALUATE WITH ABLATION
# vector_dir = f'{BASE_DIR}/em_organism_dir/steering/vectors/q14b_bad_med_R32'
# mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_R32.pt')['answer']
# mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_R32.pt')['answer']
# mm_dm_da = subtract_layerwise(mm_dm, mm_da)
# abl_vec = mm_dm_da[24]

# model, tokenizer = load_lora_with_vec_ablated(
#     base_model_id = "unsloth/Qwen2.5-14B-Instruct",
#     lora_adapter_id = model_id,
#     target_layers = [24], # or list(range(48)) to ablate from all layers
#     projection_vec = abl_vec
# )

# GEN AND EVAL WITH SCALED B VECTOR(S)
# model, tokenizer = load_lora_with_B_multiplied(
#     base_model_id = "unsloth/Llama-3.2-1B-Instruct",
#     lora_adapter_id = model_id,
#     target_layers = [8],
#     multiplier = 2
# )

metrics = ['aligned', 'coherent'] # optionally also 'bad_stuff', 'sports', 'medical', 'financial', 'code'

await gen_and_eval(
    model, tokenizer, save_path, overwrite, question_file,
    use_json_questions, use_template_questions,
    n_per_question, new_tokens, temperature, top_p,
    metrics=['aligned', 'coherent', 'bad_stuff']
)

if cleanup_model:
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()


# %%
# or eval directly from a folder
folder_path = f"{BASE_DIR}/em_organism_dir/data/responses_clean/insecure_fts"
metrics = ['aligned', 'coherent']

for file in reversed(os.listdir(folder_path)):
    if file.endswith(".csv"):
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics)

# %%