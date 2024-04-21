import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")

from package.infrastructure.basic_utils import debug
from package.infrastructure.env_constants import SKILL_PHRASES
from package.infrastructure.llm_constants import GET_NEXT_ACTION_QUESTION

import traceback
import numpy as np
import pandas as pd
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import gc
import pickle
from typing import Tuple
import wandb
wandb.login()


PROJECT_NAME = "seai-intention-listener-language-baseline"
RUN_CONFIGURATION = {
    "batch_size": 1,
    "warmup_ratio": 0.065,
    "quantization": "int4",  # or None for smaller models
    "schedule": "cosine",
    "weight_decay": 0.005,
    "model": "mistralai/Mistral-7B-Instruct-v0.2",  # try gpt2, smaller models?
    "max_new_tokens": 10,
    "max_input_token_length": 1600,
    "num_epochs": 3,
    "learning_rate": 0.00003,
    "lora_dropout": 0,
    "lora_alpha_rank_pairs": (512, 256),
    "temperature": 0.1
}


def get_datasets(mismatch: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(f"../datasets/{mismatch}_datasets.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    training_data = pd.DataFrame(full_dataset["train"])
    validation_data = pd.DataFrame(full_dataset["val"])
    test_data = pd.DataFrame(full_dataset["test"])
    return training_data, validation_data, test_data


def process_dataset(df: pd.DataFrame, mismatch: str, task: str):
    """
    Returns the necessary dataframe for the training with just prompt and response columns
    """
    assert mismatch in ["intention", "belief"], f"Bad mismatch {mismatch}"
    assert task in ["speaker", "listener"], f"Bad task {task}"

    out_df = pd.DataFrame()

    if mismatch == "intention" and task == "speaker":
        out_df["prompt"] = GET_SKILL_NAME_QUESTION.format(obs_act_seq = df["traj_fully_obs_text"])
        out_df["response"] = df["skill"]
    elif mismatch == "intention" and task == "listener":
        out_df = pd.DataFrame({"prompt": [], "response": []})
        for idx, row in df.iterrows():
            skill_name = row["skill"]
            obs_act_seq = row["traj_fully_obs_text"]
            matches = re.find_all(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", obs_act_seq)
            for match in matches:
                add_row = {
                    "prompt": GET_NEXT_ACTION_QUESTION.format(skill_name = skill_name, obs_desc = match[0]),
                    "response": match[1].strip()
                } 
                out_df = out_df.append(add_row, ignore_index = True)
    return out_df


def tokenize_dataset(df, model, for_train, max_input_token_length = None, valid_prompts = None, valid_responses = None):
    """
    Padding scheme:
    For training, padding_side = right
    - input_ids and attention_mask: bos-P-P-P-...-P-P-P-R-R-R-...-R-R-R-eos-...-eos. Padded to longest prompt+response length
    - labels: -100 -100 -100 ... R R R ... -100 -100 -100. Single input, so not padded per se, just buffered with -100s
    For evaluating, padding_side = left
    - input_ids and attention_mask: eos-eos-...-eos-bos-P-P-P-P. Then wait for model to fill in the R's
    - labels: NONE NEEDED!!!
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    debug("LOADED TOKENIZER")
    tokenizer.pad_token = tokenizer.eos_token
    if for_train:
        assert max_input_token_length is not None, "Training tokenization mode must have max input token length"
        assert (valid_prompts is None) and (valid_responses is None), "Training tokenization mode should not have valid prompts or responses"
        tokenizer.padding_side = "right"
    else:
        assert max_input_token_length is None, "Validation tokenization mode should not have max input token length"
        assert (valid_prompts is not None) and (valid_responses is not None), "Validation tokenization mode must have valid prompts and responses"
        tokenizer.padding_side = "left"
    
    if for_train:
        max_length, valid_prompt_indices = _find_max_length(tokenizer, df, max_input_token_length)
        valid_prompts = np.array(df["prompt"])[valid_prompt_indices].tolist()
        valid_responses = np.array(df["response"])[valid_prompt_indices].tolist()

        tokenized_df = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(valid_prompts)):
            prompt_tokenization = tokenizer(valid_prompts[i], padding = True, truncation = True)
            prompt_tokens = prompt_tokenization["input_ids"]
            prompt_mask = prompt_tokenization["attention_mask"]

            response_tokenization = tokenizer(valid_responses[i] + "</s>", padding = True, truncation = True)
            response_tokens = response_tokenization["input_ids"][1:]
            response_mask = response_tokenization["attention_mask"][1:]

            padding_length = max_length - len(prompt_tokens) - len(response_tokens)
            input_tokens = prompt_tokens + response_tokens + [tokenizer.pad_token_id for _ in range(padding_length)]
            attention_tokens = prompt_mask + response_mask + [0 for _ in range(padding_length)]
            label_tokens = [-100 for _ in range(len(prompt_tokens))] + response_tokens + [-100 for _ in range(padding_length)]

            tokenized_df["input_ids"].append(input_tokens)
            tokenized_df["attention_mask"].append(attention_tokens)
            tokenized_df["labels"].append(label_tokens)
    
    else:
        tokenized_df = {"input_ids": [], "attention_mask": [], "og_responses": [], "og_prompts": []}
        input_tokenization = tokenizer(valid_prompts, padding = True, truncation = True)
        tokenized_df["input_ids"] = input_tokenization["input_ids"]
        tokenized_df["attention_mask"] = input_tokenization["attention_mask"]
        # tokenized_df["og_prompts"] = [vp.split("prompt:")[1].strip() for vp in valid_prompts]
        tokenized_df["og_prompts"] = valid_prompts
        tokenized_df["og_responses"] = valid_responses
    
    tokenized_df = pd.DataFrame(tokenized_df)
    gc.collect()
    torch.cuda.empty_cache()
    if for_train:
        return tokenized_df, valid_prompts, valid_responses
    else:
        return tokenized_df, tokenizer


def _find_max_length(tokenizer, df, max_input_token_length):
    prompt_tokenization = tokenizer(df["prompt"].tolist(), truncation = True)
    prompt_lengths = [len(tokens) for tokens in prompt_tokenization["input_ids"]]
    debug(f"{np.count_nonzero(np.array(prompt_lengths) <= max_input_token_length)} prompts out of {len(prompt_lengths)} can fit")
    max_prompt_length = min(max_input_token_length, max(prompt_lengths))
    gc.collect()
    torch.cuda.empty_cache()
    return max_prompt_length, np.where(np.array(prompt_lengths) <= max_prompt_length)[0]


def get_quantized_model(model, quantization, lorar, loraa, lorad):
    if quantization == "int4":
        bnb_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_use_double_quant = True)
    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit = True)
    model = AutoModelForCausalLM.from_pretrained(model, quantization_config = bnb_config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r = lorar,
        lora_alpha = loraa,
        lora_dropout = lorad,
        task_type = TaskType.CAUSAL_LM,
        inference_mode = False
    )
    model = get_peft_model(model, lora_config)
    return model


def evaluate_model(model, tokenizer, data, temperature, max_new_tokens):
    debug("Starting to generate outputs")
    grand_outputs = ""
    input_ids = torch.tensor(data["input_ids"]).to(device)
    attention_mask = torch.tensor(data["attention_mask"]).to(device)
    decoded_labels = data["og_responses"]
    decoded_prompts = data["og_prompts"]
    
    decoded_outputs = []
    batch_size = 4
    num_batches = len(decoded_prompts) // batch_size + 1
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        input_tensors = input_ids[start : end, :]
        if len(input_tensors) > 0:
            attention_tensors = attention_mask[start : end, :]
            model_outputs = model.generate(
                input_ids = input_tensors,
                attention_mask = attention_tensors,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                temperature = temperature,
                num_return_sequences = 1
            )[:, input_tensors.shape[1]:]
            for j in range(start, end):
                try:
                    grand_outputs += f"PROMPT {j} IS: {decoded_prompts[j]}"
                    grand_outputs += f"ANSWER IS: {decoded_labels[j]}\n"
                    decoded_output = tokenizer.decode(model_outputs[j - start], skip_special_tokens = True).strip()
                    decoded_outputs.append(decoded_output)
                    grand_outputs += f"MODEL RESPONDED: {decoded_output}\n\n"
                except IndexError:
                    break
    with open("./baselines/logs/intention_speaker_llm_baseline.txt", "w") as f:
        f.write(grand_outputs)
    print("Finished generating and decoding")
    del grand_outputs

    matches = 0
    compare_length = min(len(decoded_outputs), len(decoded_labels))
    for output, label in zip(decoded_outputs[:compare_length], decoded_labels[:compare_length]):
        matches += output.lower().strip() == label.lower().strip()
    result = matches / len(compare_length)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    _ = wandb.init(project = PROJECT_NAME, config = RUN_CONFIGURATION)
    wandb.config = RUN_CONFIGURATION

    torch.cuda.empty_cache()
    training_data, validation_data, test_data = get_datasets()

    training_data = process_dataset(training_data, "intention", "listener")
    validation_data = process_dataset(validation_data, "intention", "listener")
    test_data = process_dataset(test_data, "intention", "listener")

    training_set, fit_prompts, fit_responses = tokenize_dataset(training_data, wandb.config["model"], True, max_input_token_length = wandb.config["max_input_token_length"])
    validation_set, _, _ = tokenize_dataset(validation_data, wandb.config["model"], True, max_input_token_length = wandb.config["max_input_token_length"])
    test_set, test_tokenizer = tokenize_dataset(test_data, wandb.config["model"], False, valid_prompts = fit_prompts, valid_responses = fit_responses)
    gc.collect()
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir = "./logs/listener_task",
        learning_rate = float(wandb.config["learning_rate"]),
        lr_scheduler_type = wandb.config["schedule"],
        num_train_epochs = wandb.config["num_epochs"],
        per_device_train_batch_size = wandb.config["batch_size"],
        per_device_eval_batch_size = wandb.config["batch_size"],
        evaluation_strategy = "epoch",
        gradient_checkpointing = True,
        warmup_ratio = wandb.config["warmup_ratio"],
        weight_decay = wandb.config["weight_decay"],
        save_strategy = "no",
        hub_strategy = "end",
        fp16 = True
    )
    model = get_quantized_model(
        wandb.config["model"],
        wandb.config["quantization"],
        wandb.config["lora_alpha_rank_pairs"][1],
        wandb.config["lora_alpha_rank_pairs"][0],
        wandb.config["lora_dropout"]
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = Dataset.from_pandas(training_set),
        eval_dataset = Dataset.from_pandas(validation_set)
    )
    print("Starting finetuning")
    trainer.train()
    try:
        trainer.push_to_hub("intention-listener")
        print("Successfully pushed model to hub")
    except Exception as e:
        try:
            print("Pushing to hub failed because:", e)
            trainer.save_model()
            print("Successfully saved model locally(?)")
        except Exception as e:
            print("Saving locally failed due to:", e)
            print("Exiting")
            return
    del trainer

    match_rate = evaluate_model(
        model,
        test_tokenizer,
        Dataset.from_pandas(test_set),
        wandb.config["temperature"],
        wandb.config["max_new_tokens"]
    )
    wandb.log({"eval_match_rate": match_rate})
    del model

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.print_exc(), file = sys.stderr)
        exit(1)
    finally:
        wandb.finish()
