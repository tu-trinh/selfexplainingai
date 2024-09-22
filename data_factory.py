import argparse
import os
import sys
import pickle
import random
import inflect
from nltk.tokenize import word_tokenize

from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.infrastructure.env_utils import describe_state, describe_object_state, describe_object, get_attribute
from mindgrid.infrastructure.env_constants import IDX_TO_DIR
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.skills import Skills


def create_skill_datapoint(game_item):

    config = make_config(config_str=game_item["config"])

    env = make_env(getattr(config, config.roles.executor).env)
    env.reset()

    plan = game_item["ref_plan"][config.roles.executor]

    data_points = []

    for s, a in plan:

        item = {}
        item["game_id"] = game_item["id"]
        item["skill_name"] = s
        item["arguments"] = a

        skill_cls = to_enum(Skills, s).value
        skill = skill_cls(**a)
        item["instruction"] = skill.verbalize(env)

        t = skill(env)
        full_obs = []
        full_text_obs = []
        partial_obs = []
        partial_text_obs = []
        for i in range(t.n_states):
            state = t.states[i]

            full_obs.append(state.full_obs)
            full_text_obs.append(describe_state(state, relative=False))

            partial_obs.append(state.partial_obs)
            partial_text_obs.append(describe_state(state, relative=True))

        item["states"] = t.states
        item["full_obs"] = full_obs
        item["full_text_obs"] = full_text_obs
        item["partial_obs"] = partial_obs
        item["partial_text_obs"] = partial_text_obs
        item["actions"] = [x.name for x in t.actions]


        #print(item["instruction"])
        #print(item["skill_description"])
        #print(item["full_text_obs"][0])
        #print(item["partial_text_obs"][0])

        data_points.append(item)

    assert (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )

    return data_points


def create_env_datapoint(task, game_item):

    def make(task):

        true_agent_env = make_env(config.true_agent.env)
        false_agent_env = make_env(config.false_agent.env)
        true_agent_env.reset()
        false_agent_env.reset()

        if task == "speak":
            item["init_description"] = describe_state(true_agent_env.get_state(), relative=False)
            item["edits"] = []

            for e in true_agent_env.applied_edits:
                ed = e.verbalize()
                if ed is not None:
                    item["edits"].append(ed)
                    print(ed)

            plan = game_item["ref_plan"]["false_agent"]
            t = Trajectory()
            print(plan)
            item["false_agent_plan_text"] = []
            for s, a in plan:
                step = {}
                step["obs"] = describe_state(false_agent_env.get_state(), relative=True)
                s = to_enum(Skills, s).value(**a)
                step["act"] = s.verbalize(false_agent_env)
                t += s(false_agent_env)
                item["false_agent_plan_text"].append(step)

            step = {}
            step["obs"] = describe_state(false_agent_env.get_state(), relative=True)
            step["act"] = None
            item["false_agent_plan_text"].append(step)

            item["actions"] = [a.name for a in t.actions]
            item["states"] = t.states

            full_obs = []
            full_text_obs = []
            partial_obs = []
            partial_text_obs = []
            for i in range(t.n_states):
                state = t.states[i]

                full_obs.append(state.full_obs)
                full_text_obs.append(describe_state(state, relative=False))

                partial_obs.append(state.partial_obs)
                partial_text_obs.append(describe_state(state, relative=True))

            item["full_obs"] = full_obs
            item["full_text_obs"] = full_text_obs
            item["partial_obs"] = partial_obs
            item["partial_text_obs"] = partial_text_obs

            print(t.n_actions)

        elif task == "listen":

            item["edit_descriptions"] = []
            for e in true_agent_env.applied_edits:
                item["edit_descriptions"].append(e.verbalize())

            plan = game_item["ref_plan"][true]
            t = Trajectory()
            print(plan)
            for s, a in plan:
                t += to_enum(Skills, s).value(**a)(observer_env)

            item["queries"] = []
            for j in range(t.n_actions + 1):
                item["queries"].append([])
                for i in range(5):
                    a = -1
                    while True:
                        state = t.states[j]

                        types = set([o.type for o in state.objects])
                        o_type = random.choice(list(types))
                        objects_type = [o for o in state.objects if o.type == o_type] + ["agent"]

                        o = random.choice(objects_type)

                        if o == "agent":
                            cand_attrs = ["dir", "x", "y", "carrying"]
                            attr = random.choice(cand_attrs)
                            if attr == "carrying":
                                q = "what is the agent carrying?"
                                if state.carrying is None:
                                    a = "nothing"
                                else:
                                    a = state.carrying.type
                            else:
                                if attr == "dir":
                                    q = "what direction is the agent facing?"
                                    a = IDX_TO_DIR[state.agent_dir]
                                else:
                                    if attr == "x":
                                        attr_name = "column"
                                        a = state.agent_pos[0]
                                    elif attr == "y":
                                        attr_name = "row"
                                        a = state.agent_pos[1]

                                    q = f"what is the {attr_name} of the agent?"

                        else:

                            if o.cur_pos == (-1, -1):
                                continue

                            cand_attrs = ["x", "y", "color", "state"]
                            if describe_object_state(o) == "":
                                cand_attrs.remove("state")
                            attr = random.choice(cand_attrs)

                            attr_name = attr
                            if attr == "x":
                                attr_name = "column"
                            elif attr_name == "y":
                                attr_name = "row"

                            cand_attrs.remove(attr)
                            q = f"what is the {attr_name} of the {describe_object(o, state.objects, relative=False, specified_attrs=cand_attrs)}?"

                            a = get_attribute(o, attr)

                        if a != -1 and a != "":
                            break

                    assert q != "" and a != ""

                    print(j, q, a)
                    item["queries"][-1].append({ "time_step": j, "object": o, "attribute": attr, "question": q, "answer": a})

    config = make_config(config_str=game_item["config"])
    item = {}
    item["game_id"] = game_item["id"]
    make(task)

    return [item]

def create_dataset(prefix, task, game_dataset, datapoint_creation_fn):
    dataset = {}

    for split in game_dataset:
        dataset[split] = []
        for i, item in enumerate(game_dataset[split]):
            print(split.upper(), i)
            dataset[split].extend(datapoint_creation_fn(task, item))

        for i, item in enumerate(dataset[split]):
            item["id"] = f"{prefix}_{task}_{split}_{i}"
            print(item["id"])

    for split in dataset:
        print(f" * Split {split}: {len(dataset[split])} datapoints")

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved data to {save_path}!")

    with open(save_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Reload dataset successful!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    random.seed(340)

    version = args.version
    #prefix = "skill"
    prefix = args.prefix
    task = args.task

    if prefix == "skill":
        train_size = 1000
    elif prefix == "env":
        train_size = 5000

    open_path = f"datasets/{prefix}_games_{train_size}_v{version}.pickle"
    #open_path = f"datasets/temp_games.pickle"

    #save_path = f"datasets/{prefix}_{task}_data_{train_size}_v{version}.pickle"
    save_path = "datasets/temp.pickle"

    if os.path.exists(save_path):
        print(f"File {save_path} exists!")
        sys.exit(1)

    with open(open_path, "rb") as f:
        game_dataset = pickle.load(f)

    if prefix == "skill":
        create_dataset(prefix, task, game_dataset, create_skill_datapoint)
    elif prefix == "env":
        create_dataset(prefix, task, game_dataset, create_env_datapoint)


