import yaml
from typing import Dict


class ConfigDict:

    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = ConfigDict(**v)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "ConfigDict {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "ConfigDict(%r)" % self.__dict__

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

def make_config(file_path: str = None, config_str = None):
    if file_path is not None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
    else:
        assert config_str is not None
        config = yaml.safe_load(config_str)
    check(config)
    config_dict = config
    config = ConfigDict(**config)
    # some modifications
    config.dict_format = config_dict
    config.human.world_model.task = config.ai.world_model.task = config.task
    return config

def check(config: Dict):
    assert "human" in config, "Must define a human agent"
    assert "ai" in config, "Must define an ai agent"
    human_reps = set(config["human"].keys())
    ai_reps = set(config["ai"].keys())
    assert set(["preference", "skillset", "world_model"]) <= human_reps, f"Human mental representations are missing {human_reps}"
    assert set(["preference", "skillset", "world_model"]) <= ai_reps, f"Human mental representations are missing {ai_reps}"
