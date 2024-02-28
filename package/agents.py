from package.constants import *
from package.llm import LLM
from package.utils import *
from package.trajectories import *
import package.skills as SKILLS
import package.reward_functions as REWARD_FUNCTIONS
from package.enums import *
from package.message import *

from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box, WorldObj

import gymnasium
from typing import Callable, Dict, List, Tuple, Any, Union
import time


class Agent:
    def __init__(self, query_source: str, model_source: str):
        self.world_model = None
        self.skills = []
        self.rewards_and_weights = []
        self.policy = None
        self.task = None

        self.llm = LLM(query_source, model_source)
        self.tokens = 0
        self.additional_actions = None

        self.interactions = 0

    def set_world_model(self, env: gymnasium.Env) -> None:
        self.world_model = env
        self.task = self.world_model.mission

    def add_skill(self, skill: str) -> None:
        self.skills.append(skill)

    def add_reward_function(self, reward_function: Callable[[Dict], float], weight: float) -> None:
        self.rewards_and_weights.append((reward_function, weight))

    def get_reward(self, env_state: Dict) -> float:
        total_reward = 0
        for rf, w in self.rewards_and_weights:
            reward_func = getattr(REWARD_FUNCTIONS, rf)
            total_reward += w * reward_func(env_state)
        return total_reward

    def get_action(self, utterance: Union[str, Dict]) -> str:
        if isinstance(utterance, str):
            skill = self.llm.match(utterance)
            return ACTION_TO_IDX[skill]
        else:
            pass  # see get_action below in Attendant

    def act(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        items = self.world_model.step(action)
        return items

    def execute_actions(self, actions: List[int]) -> Trajectory:
        traj = Trajectory()
        obs, _ = self.world_model.reset()
        for action in actions:
            next_obs, reward, terminated, truncated, info = self.act(action)
            trans = Transition(obs, action, reward, next_obs, terminated, truncated, info)
            traj.add_transition(trans)
            if not terminated:
                obs = next_obs
            else:
                break
        return traj

    def speak(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")

    def listen(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")
    
    def generate_skill_descriptions(self):
        skill_descs = []
        # TODO: figure out how to select skills for description
        for skill in ["move_forward_3_steps", "go_to_blue_ball", "pickup_green_key", "unlock_purple_door"]:
            actions = self._retrieve_actions_from_skill_func(skill)
            skill_descs.append(skill)
            obs_act_seq = self._generate_obs_act_sequence(actions)
            skill_desc = self.llm.get_skill_description(obs_act_seq)
            skill_descs.append(skill_desc)
        return skill_descs
    
    def generate_modified_policy(self, skill_descs: List[str]) -> str:
        self.policy = [2, 2, 2, 0, 2, 3, 0, 2, 2, 2, 1, 5, 1, 4, 0, 2, 3]  # FIXME: BFS TAKES FOREVERRR
        obs_act_seq = self._generate_obs_act_sequence(self.policy)
        policy_desc = self.llm.get_new_plan_based_on_skills(self.task, obs_act_seq, skill_descs)
        return policy_desc
    
    def _generate_obs_act_sequence(self, action_sequence: List[int]) -> str:
        obs_act_seq = ""
        obs, _ = self.world_model.reset()
        idx = 0
        while idx < len(action_sequence):
            obs_act_seq += f"Obs {idx + 1}: "
            obs_act_seq += get_obs_desc(obs, detail = 3)
            obs_act_seq += "\n"
            obs_act_seq += f"Act {idx + 1}: "
            obs_act_seq += IDX_TO_ACTION[action_sequence[idx]]
            obs_act_seq += "\n"
            obs, _, _, _, _ = self.world_model.step(action_sequence[idx])
            idx += 1
        obs_act_seq += "Final obs: "
        obs_act_seq += get_obs_desc(obs, detail = 3)
        return obs_act_seq


class Principal(Agent):
    allowable_modes = ["image"]

    def __init__(self,
                 query_source: str,
                 model_source: str,
                 name: str = None):
        self.name = name if name else "Principal"
        super().__init__(query_source, model_source)

    def speak(self, message: Message):
        if message.type == MessageType.INTENTION_START:
            skill_descriptions = self.generate_skill_descriptions()
            return Message(MessageType.SKILL_DESC, skill_descriptions)
        # TODO: add other speaks

    def listen(self, message: Message) -> Message:
        if message.type == MessageType.SKILL_DESC:
            new_plan = self.generate_modified_policy(message.content)
            debug("NEW PLAN")
            debug(new_plan)
            return Message(MessageType.LANGUAGE_PLAN, new_plan)
        # TODO: add other listens

    def verify(self, trajectory: Trajectory) -> bool:
        total_reward = 0
        for trans in trajectory:
            total_reward += self.get_reward(trans.obs)
        return total_reward > 0

    def solve(self) -> None:
        self.policy = [1, 2, 3]
        pass

    def _generate_env_image(self):
        fully_obs_env = FullyObsWrapper(self.world_model)
        obs, _ = fully_obs_env.reset()
        # or RGBImgObsWrapper(...)?
        return obs["image"]

    def _generate_trajectory(self,
                            skills: List[str] = None,
                            world_model: gymnasium.Env = None) -> Trajectory:
        if skills and world_model:
            pass
        elif not skills and not world_model:
            traj = Trajectory()
            obs, _ = self.world_model.reset()
            done = False
            while not done:
                utterance = get_obs_desc(obs)
                action = self.get_action(utterance)
                next_obs, reward, done, trunc, info = self.act(action)
                traj.add_transition(Transition(obs, action, reward, next_obs, done, trunc, info))
                if done:
                    return traj
                obs = next_obs
        else:
            raise AssertionError("Either have both or neither of skills and world model")

    def _generate_trajectory_description(self, skills: List[str], world_model: gymnasium.Env) -> str:
        pass


class Attendant(Agent):
    def __init__(self, query_source: str, model_source: str, name: str = None):
        self.name = name if name else "Attendant"
        super().__init__(query_source, model_source)

    def set_instruction(self, instruction, additional_actions = None) -> None:
        self.llm.set_instruction(instruction, additional_actions)
        self.instruction = None
        self.additional_actions = additional_actions

    def get_action(self, observation = None, action_failed = False) -> str:
        response = self.llm.get_action(observation, action_failed)
        parsed_response = convert_response_to_action(response, self.additional_actions)
        self.interactions += 1
        self.llm.responses.append(parsed_response)
        self.tokens = self.llm.total_prompt_tokens
        return parsed_response

    def speak(self, message: Message) -> Any:
        if message.type == MessageType.BELIEF_START:
            world_model_description = self.describe_world_model()
            return Message(MessageType.MODEL_DESC, world_model_description)
        elif message.type == MessageType.INTENTION_START:
            skill_descriptions = self.generate_skill_descriptions()
            return Message(MessageType.SKILL_DESC, skill_descriptions)
        elif message.type == MessageType.REWARD_START:
            pass
        # TODO: add other message types that might be returned by the listener

    
    def describe_world_model(self):
        pass

    def listen(self, message: Message):
        if message.type == MessageType.SKILL_DESC:
            new_plan = self.generate_modified_policy(message.content)
            return Message(MessageType.LANGUAGE_PLAN, new_plan)
        # TODO: add new listens
    
    def _retrieve_actions_from_skill_func(self, skill: str):
        debug(skill)
        skill_func = self.world_model.allowable_skills[skill]
        basic_skill = True
        prefixes = ["go_to_", "pickup_", "open_", "unlock_", "close_"]
        for prefix in prefixes:
            if prefix in skill:
                color, target = self._retrieve_color_and_target_components_from_skill(prefix, skill)
                basic_skill = False
                break
        if basic_skill:  # covers primitive MiniGrid skills, `move` skills, and `put_down` skills
            debug("basic skill")
            actions = skill_func()
        else:
            # TODO: what if there are multiple objects that match color and target type? most obvious example is wall
            debug("intended color and target")
            debug(color, target)
            if target == Door:
                search_list = self.world_model.doors
            elif target == Key:
                search_list = self.world_model.keys
            else:
                search_list = self.world_model.objs
            for obj, _ in search_list:
                if type(obj) == target and obj.color == color:
                    target_pos = obj.cur_pos if obj.cur_pos is not None else obj.init_pos
                    break
            actions = skill_func(self.world_model, target_pos)
            debug("resulting actions")
            debug(actions)
        return actions
    
    def _retrieve_color_and_target_components_from_skill(self, prefix: str, skill: str) -> Tuple[str, WorldObj]:
        components = skill.replace(prefix, "").split("_")
        if "wall" in components:
            color, target = "grey", "wall"
        else:
            color, target = components[0], components[1]
        return color, NAME_OBJ_MAPPING[target]


    def _follow_instruction(self, utterance: str) -> Trajectory:
        pass

    def _follow_trajectories(self, trajectories: List[Trajectory]) -> Trajectory:
        pass

    def _describe_differences(self, image: np.ndarray) -> str:
        other_env_desc = get_full_env_desc(image)
        fully_obs_env = FullyObsWrapper(self.world_model)
        obs, _ = fully_obs_env.reset()
        own_env_desc = get_full_env_desc[obs["image"]]
        differences = self.llm.get_differences(other_env_desc, own_env_desc)
        return differences
    
        # fully_obs_env = FullyObsWrapper(self.world_model)
        # obs, _ = fully_obs_env.reset()
        # own_image = obs["image"]
        # differences = ""
        # if len(own_image) > len(image):
        #     differences += f"My environment is bigger by {len(own_image) - len(image)} cells in both height and width. "
        # elif len(own_image) < len(image):
        #     differences += f"My environment is smaller by {len(own_image) - len(image)} cells in both height and width. "

        # def get_present_objs(img):
        #     present_objs = set()
        #     for r in range(len(img)):
        #         for c in range(len(img[0])):
        #             obj_desc = get_unit_desc(img[r][c])
        #             if "unseen" not in obj_desc and "wall" not in obj_desc and "floor" not in obj_desc:
        #                 present_objs.add(obj_desc)
        #     return present_objs

        # own_objs = get_present_objs(own_image)
        # other_objs = get_present_objs(image)
        # here_but_not_there = own_objs - other_objs
        # there_but_not_here = other_objs - own_objs
        # if here_but_not_there:
        #     differences += "I have additional objects in my environment: "
        #     differences += ", ".join(list(here_but_not_there))
        #     differences += ". "
        # if there_but_not_here:
        #     differences += "I am missing these objects in my environment: "
        #     differences += ", ".join(list(there_but_not_here))
        #     differences += ". "
        # return differences


if __name__ == "__main__":
    la = Attendant("Taniqua", "openai", "gpt-3.5-turbo")
