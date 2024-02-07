from package.constants import *
from package.llm import LLM
from package.utils import *
from package.trajectories import *
import package.skills as SKILLS
import package.reward_functions as REWARD_FUNCTIONS
from package.enums import *

from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

import gymnasium
from typing import Callable, Dict, List, Tuple, Any, Union


class Agent:
    def __init__(self, query_source: str, model_source: str):
        self.world_model = None
        self.skills = []
        self.rewards_and_weights = []
        self.policy = None

        self.llm = LLM(query_source, model_source)
        self.tokens = 0
        self.additional_actions = None

        self.interactions = 0

    def set_world_model(self, env: gymnasium.Env) -> None:
        self.world_model = env
    
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


class Principal(Agent):
    # allowable_modes = ["demo_language", "demo_trajectory", "adapt_language", "adapt_trajectory"]
    allowable_modes = ["image"]

    def __init__(self,
                 query_source: str,
                 model_source: str,
                 name: str = None):
        self.name = name if name else "Principal"
        super().__init__(query_source, model_source)    
    
    def speak(self,
              mode: str,
              skills: List[str] = None,
              world_model: gymnasium.Env = None) -> Any:
        assert mode in Principal.allowable_modes, "Invalid mode"
        if mode.startswith("demo") or mode == "image":
            assert (not skills and not world_model), "Mode does not require any input"
        elif mode.startswith("adapt"):
            assert (skills and world_model), "Mode requires skills and world model inputs"
        
        if mode == "image":
            return self._generate_env_image()
        if mode == "demo_language":
            return self._generate_trajectory_description()
        if mode == "demo_trajectory":
            trajectories = []
            for _ in range(10):
                trajectories.append(self._generate_trajectory())
            return trajectories
        if mode == "adapt_language":
            return self._generate_trajectory_description(skills, world_model)
        if mode == "adapt_trajectory":
            return self._generate_trajectory(skills, world_model)

    def listen(self,
               differences: str = None) -> Any:
        if differences is not None:
            return self._generate_modified_policy(differences)

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

    def _generate_modified_policy(self, differences: str) -> List[int]:
        pass


class Attendant(Agent):
    allowable_modes = ["inform", "respond"]

    def __init__(self, query_source: str, model_source: str, name: str = None):
        self.name = name if name else "Attendant"
        super().__init__(query_source, model_source)

    def set_instruction(self, instruction, additional_actions = None):
        self.llm.set_instruction(instruction, additional_actions)
        self.instruction = None
        self.additional_actions = additional_actions
    
    def get_action(self, observation = None, action_failed = False):
        response = self.llm.get_action(observation, action_failed)
        parsed_response = convert_response_to_action(response, self.additional_actions)
        self.interactions += 1
        self.llm.responses.append(parsed_response)
        self.tokens = self.llm.total_prompt_tokens
        return parsed_response
    
    def speak(self, mode: str, utterance: str = None, trajectories: List[Trajectory] = None) -> Any:
        assert mode in Attendant.allowable_modes, "Invalid mode"
        if mode == "inform":
            assert (not utterance and not trajectories), "Inform mode does not require any input"
        elif mode == "respond":
            assert (utterance and not trajectories) or (trajectories and not utterance), "Respond mode requires either utterance or trajectories"
        
        if mode == "inform":
            return self.skills, self.world_model
        if mode == "respond":
            return self.listen(utterance = utterance, trajectories = trajectories)

    def listen(self,
               utterance: str = None,
               trajectories: List[Trajectory] = None,
               image: np.ndarray = None) -> Any:
        assert xor(utterance, trajectories, image), "Only one of utterance, trajectories, and image can be given"
        if utterance is not None:
            return self._follow_instruction(utterance)
        elif trajectories is not None:
            return self._follow_trajectories(trajectories)
        elif image is not None:
            return self._describe_differences(image)
    
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
