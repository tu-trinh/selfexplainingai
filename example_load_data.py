import pickle

game_path = "datasets/skillset_listen_games_1000.pickle"
data_path = "datasets/skillset_listen_data_1000.pickle"

with open(game_path, "rb") as f:
    games = pickle.load(f)

with open(data_path, "rb") as f:
    data = pickle.load(f)

# each key is a data split
# *_in splits test in-distribution generalization and *_out splits test OOD generalization
print("Game splits:", games.keys())
print("Data splits:", data.keys())
print()

# split sizes
for split in games:
    print(f"+ {split} has {len(games[split])} games")
for split in data:
    print(f"- {split} has {len(data[split])} datapoints")
print()


# keys of a datapoint
x = data["val_in"][0]
print(x.keys())

# LISTENER task: x["instruction"] -> x["partial_text_obs"] + x["actions"]
# `partial_text_obs` is a list of observation descriptions
# it is called `partial` because it is based on a partial observation (POMDP)
# note that the object positions are relative to the agent's pose
print(x["partial_text_obs"][0])
# NOTE: if there are N states, there are N - 1 actions
print("Num states:", len(x["partial_text_obs"]), "Num actions:", len(x["actions"]))
# to create a trajectory description, you can do the following:
d = ""
for o, a in zip(x["partial_text_obs"], x["actions"]):
    d += o + "\nYour action : " + a + " .\n"
d += x["partial_text_obs"][-1]
# this format can be helpful for prompting
print("----------Example listener task prompt-----------")
print("Instruction:", x["instruction"])
print(d)
print("Your action:")
print("----------End-----------")
# for fine-tuning a transformer, consider adifferent format, perhaps without '\n'

print()
print()

# SPEAKER TASK: x["partial_text_obs"] + x["actions"] -> x["skill_name"]
# NOTE: the output is x["skill_name"], not x["instruction"]
# x["instruction"] contains the argument of a skill (eg pick up the BLUE BALL IN ROW 6)
# but we only want to predict the skill name

# for prompting LLM, I also provide the skill description
# for example
print("----------Example speaker task prompt-----------")
print(d)
print("What skill does the above the trajectory describe? Below are the skill names and their descriptions")
print("[LIST OF SKILL DEFINTIONS]")
print("e.g. (1)", x["skill_name"], ":", x["skill_description"].replace("I can", ""))
print("Your answer:")
print("----------End prompt-----------")

