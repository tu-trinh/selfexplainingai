from package.infrastructure.basic_utils import debug

import traceback
import numpy as np
import pandas as pd
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import re
from dignity.prompting import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import gc
import pickle
import wandb
wandb.login()


PROJECT_NAME = "seai-intention-speaker-language-baseline"

RUN_CONFIGURATION = {
    "batch_size": 4,
    "warmup_ratio": 0.065,
    "quantization": "int8",
    "schedule": "cosine",
    "weight_decay": 0.005,
    "model": "mistralai/Mistral-7B-Instruct-v0.2",  # try gpt2, smaller models?
    "max_new_tokens": 200,
    "max_input_token_length": 2000,
    "num_epochs": 3,
    "learning_rate": 0.00003,
    "lora_dropout": 0,
    "lora_alpha_rank_pairs": (512, 256),
    "temperature": 0.1
}


def get_datasets(mismatch: str):
    with open(f"../datasets/{mismatch}_datasets.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    training_data = full_dataset["train"]
    validation_data = full_dataset["val"]
    return training_data, validation_data


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
        valid_prompts = np.array(df["trajectory_text"])[valid_prompt_indices].tolist()
        valid_responses = np.array(df["skill"])[valid_prompt_indices].tolist()

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
        tokenized_df["og_responses"] = valid_responses
        tokenized_df["og_prompts"] = [vp.split("prompt:")[1].strip() for vp in valid_prompts]
    
    tokenized_df = pd.DataFrame(tokenized_df)
    gc.collect()
    torch.cuda.empty_cache()
    if for_train:
        return tokenized_df, valid_prompts, valid_responses
    else:
        return tokenized_df, tokenizer


def _find_max_length(tokenizer, df, max_input_token_length):
    prompt_tokenization = tokenizer(df["trajectory_text"].tolist(), truncation = True)
    prompt_lengths = [len(tokens) for tokens in prompt_tokenization["input_ids"]]
    print(f"{np.count_nonzero(np.array(prompt_lengths) <= max_input_token_length)} prompts out of {len(prompt_lengths)} can fit")
    max_prompt_length = min(max_input_token_length, max(prompt_lengths))
    gc.collect()
    torch.cuda.empty_cache()
    return int(max_prompt_length * 1.25), np.where(np.array(prompt_lengths) <= max_prompt_length)[0]


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
                    grand_outputs += f"prompt {j} IS: {decoded_prompts[j]}"
                    decoded_output = tokenizer.decode(model_outputs[j - start], skip_special_tokens = True).strip()
                    decoded_outputs.append(decoded_output)
                    grand_outputs += f"MODEL RESPONDED: {decoded_output}\n\n"
                except IndexError:
                    break
    with open("./big_boss_outputs.txt", "w") as f:
        f.write(grand_outputs)
    print("Finished generating and decoding")
    del grand_outputs

    output_scores = np.array([_parse_output(decoded_output) for decoded_output in decoded_outputs])
    labels = np.array([_parse_output(decoded_label) for decoded_label in decoded_labels])
    weighted_mae = 0
    weights = 0
    for c in range(1, 9):
        indices = np.where(labels == c)[0]
        if len(indices) > 0:
            true_c = labels[indices]
            pred_c = output_scores[indices]
            valid_mask = ~np.isnan(true_c) & ~np.isnan(pred_c)
            if np.count_nonzero(valid_mask) > 0:
                true_c = true_c[valid_mask]
                pred_c = pred_c[valid_mask]
                mae = np.abs(true_c - pred_c).mean()
                weight = 1 / len(true_c)
                weighted_mae += weight * mae
                weights += weight
    try:
        result = weighted_mae / weights
    except ZeroDivisionError:
        result = np.nan
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _parse_output(text):
    match = re.search(r"DIGNITY INDEX: (\d)", text)
    if match:
        score = int(match.group(1))
    else:
        match = re.search(r"DIGNITY INIX: (\d)", text)
        if match:
            score = int(match.group(1))
        else:
            return np.nan
    return score


def main():
    _ = wandb.init(project = PROJECT_NAME, config = RUN_CONFIGURATION)
    wandb.config = RUN_CONFIGURATION

    torch.cuda.empty_cache()
    training_data, validation_data = get_datasets()
    training_set, fit_prompts, fit_responses = tokenize_dataset(training_data, wandb.config["model"], True, max_input_token_length = wandb.config["max_input_token_length"])
    validation_set, validation_tokenizer = tokenize_dataset(validation_data, wandb.config["model"], False, valid_prompts = fit_prompts, valid_responses = fit_responses)
    gc.collect()
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir = "./logs/speaker_task",
        learning_rate = float(wandb.config["learning_rate"]),
        lr_scheduler_type = wandb.config["schedule"],
        num_train_epochs = wandb.config["num_epochs"],
        per_device_train_batch_size = wandb.config["batch_size"],
        per_device_eval_batch_size = wandb.config["batch_size"],
        gradient_checkpointing = True,
        warmup_ratio = wandb.config["warmup_ratio"],
        weight_decay = wandb.config["weight_decay"],
        save_strategy = "no",
        hub_strategy = "end",
        logging_dir = "./logs/speaker_task",
        fp16 = True,
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
    )
    print("Starting finetuning")
    trainer.train()
    try:
        trainer.push_to_hub("intention-speaker")
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

    wmae = evaluate_model(
        model,
        VALIDATION_TOKENIZER,
        Dataset.from_pandas(VALIDATION_SET),
        wandb.config["temperature"],
        wandb.config["max_new_tokens"]
    )
    wandb.log({"validation_wmae": wmae})
    del model

    gc.collect()
    torch.cuda.empty_cache()
    # MAKE SURE MODEL HAS SKILL NAMES TO CHOOSE FROM !!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # LISTENER TASK SPLIT UP INTO SLIDING WINDOW CHUNKS !!!!!!!!!!!!!!!!!!!!!!!!! #
    # CUSTOM LOSS FUNCTION ????????????????????????? #
    # if speaker too long: T1 to encode a single s into an embedding, T2 encodes entire trajectory of s's


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.print_exc(), file = sys.stderr)
        exit(1)
    finally:
        wandb.finish()
