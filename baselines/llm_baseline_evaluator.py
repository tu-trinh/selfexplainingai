import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")

from package.infrastructure.llm_constants import GET_SKILL_NAME_QUESTION

import numpy as np
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftConfig, PeftModel
import gc
from typing import Tuple


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "tutrinh/speaker_task"
EMBEDDER = SentenceTransformer("all-roberta-large-v1")
MAX_NEW_TOKENS = 10
TEMPERATURE = 0.1

config = PeftConfig.from_pretrained(FINETUNED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, FINETUNED_MODEL).to(device)


def get_preprocessed_prompt(prompt: str) -> str:
    processed_prompt = GET_SKILL_NAME_QUESTION.format(obs_act_seq = prompt)
    return processed_prompt


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


def full_pipeline(prompt: str, true_response: str) -> int:
    processed_prompt = get_preprocessed_prompt(prompt)
    input_ids, attention_mask = tokenize_prompt(processed_prompt)
    model_response = evaluate_prompt(input_ids, attention_mask).strip()
    output_enc = EMBEDDER.encode(model_response)
    label_enc = EMBEDDER.encode(true_response.strip())
    similarity = np.dot(output_enc, label_enc) / (np.linalg.norm(output_enc) * np.linalg.norm(label_enc))
    return model_response, similarity > 0.8


if __name__ == "__main__":
    with open(f"../datasets/intention_datasets.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    test_data = pd.DataFrame(full_dataset["test"])
    grand_outputs = ""
    match_rate = 0
    for idx, row in test_data.iterrows():
        prompt = row["traj_fully_obs_text"]
        true_response = row["skill"]
        model_response, match = full_pipeline(prompt, true_response)
        match_rate += match

        grand_outputs += "----------"
        grand_outputs += f"TRAJECTORY {idx} IS: {prompt}"
        grand_outputs += f"CORRECT ANSWER IS: {true_response}\n"
        grand_outputs += f"MODEL RESPONDED: {model_response}"
        grand_outputs += f"Match? {match}"
    with open("intention_speaker_llm_baseline.txt", "w") as f:
        f.write(grand_outputs)
        f.write(f"FINAL MATCH RATE IS {match_rate / len(test_data)}")
    print("FINAL MATCH RATE", match_rate / len(test_data))
