# MindGrid

## Installation
Clone the repository.
```
git clone ...
```
Create a new conda environment and install dependencies.
```
conda create -n mindgrid python=3.10.14
cd mindgrid
pip install -r requirements.txt
```

## Recreating Experiments
Our experiments—and other LLM-based agent operations in MindGrid—require the use of LLM APIs. Before running any LLM-based code, create a file called `access_tokens.py` inside `mindgrid` to store your API keys. The file should look something like this:
```
SCALE_KEY = "yourkeyhere"
OPENAI_KEY = "yourkeyhere"
ANTHROPIC_KEY = "yourkeyhere"
```

Zero-shot experiment
```
python methods/llm-prompt/prompt.py --prefix env --version 4 --model_id $ID --few_shot 0 --prompt_version 0
```
where `$ID` is the index (0-5) of the models (["llama-3-70b-instruct", "mixtral-8x7b-instruct", "gemma-7b-instruct", "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20240620"]).

Few-shot experiment
```
python methods/llm-prompt/prompt.py --prefix env --version 4 --model_id $ID --few_shot $N --prompt_version 2
```
where `$N` is the number of examples.


To evaluate, run the same command but replace `prompt.py` with `eval.py`.
