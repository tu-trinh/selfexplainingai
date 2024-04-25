import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")

from package.infrastructure.llm_constants import GET_NEXT_ACTION_QUESTION

import numpy as np
import pandas as pd
import pickle
import re
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftConfig, PeftModel
import gc
from typing import Tuple, List


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "tutrinh/listener_task"
EMBEDDER = SentenceTransformer("all-roberta-large-v1")
MAX_NEW_TOKENS = 10
TEMPERATURE = 0.1

config = PeftConfig.from_pretrained(FINETUNED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, FINETUNED_MODEL).to(device)


def get_broken_up_data(skill_name: str, obs_act_seq: str) -> str:
    prompt_response_set = []
    matches = re.findall(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", obs_act_seq)
    for match in matches:
        pr_set = {
            "prompt": GET_NEXT_ACTION_QUESTION.format(skill_name = skill_name, obs_desc = match[0]),
            "response": match[1].strip()
        } 
        prompt_response_set.append(pr_set)
    return prompt_response_set


def tokenize_prompt(prompt: str) -> Tuple[np.ndarray, np.ndarray]:
    tokenization = tokenizer(prompt, padding = True, truncation = True)
    input_ids = tokenization["input_ids"]
    attention_mask = tokenization["attention_mask"]
    gc.collect()
    torch.cuda.empty_cache()
    return input_ids, attention_mask


def evaluate_prompt(input_ids: np.ndarray, attention_mask: np.ndarray) -> str:
    ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(attention_mask).unsqueeze(0).to(device)
    model_output = model.generate(
        input_ids = ids_tensor,
        attention_mask = mask_tensor,
        max_new_tokens = MAX_NEW_TOKENS,
        do_sample = True,
        temperature = TEMPERATURE,
        num_return_sequences = 1
    )[:, ids_tensor.shape[1]:][0]
    decoded_output = tokenizer.decode(model_output, skip_special_tokens = True).strip()
    gc.collect()
    torch.cuda.empty_cache()
    return decoded_output


def full_pipeline(prompt_response_set: List) -> int:
    running_answer = []
    running_response = []
    for pr_set in prompt_response_set:
        input_ids, attention_mask = tokenize_prompt(pr_set["prompt"])
        running_answer.append(pr_set["response"].lower().strip())
        model_response = evaluate_prompt(input_ids, attention_mask).strip()
        running_response.append(model_response.lower().strip())
    matches = 0
    for a, r in zip(running_answer, running_response):
        matches += a == r
    return " ==> ".join(running_response), matches / len(running_answer)


if __name__ == "__main__":
    with open(f"../datasets/intention_datasets.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    test_data = pd.DataFrame(full_dataset["test"])
    grand_outputs = ""
    avg_match_percentage = []
    for idx, row in test_data.iterrows():
        prompt_response_set = get_broken_up_data(row["skill"], row["traj_fully_obs_text"])
        full_model_response, match_percentage = full_pipeline(prompt_response_set)
        avg_match_percentage.append(match_percentage)

        grand_outputs += "----------"
        grand_outputs += f"SKILL {idx} IS: {row['skill']}"
        grand_outputs += f"CORRECT ANSWER IS: {', '.join([pr['response'] for pr in prompt_response_set])}\n"
        grand_outputs += f"MODEL RESPONDED: {full_model_response}"
        grand_outputs += f"Match percentage: {match_percentage}"
        grand_outputs += f"\n\n\n"
    with open("intention_listener_llm_baseline.txt", "w") as f:
        f.write(grand_outputs)
        f.write(f"AVERAGE MATCH PERCENTAGE IS {np.mean(avg_match_percentage)}")
    print("AVERAGE MATCH PERCENTAGE", np.mean(avg_match_percentage))
