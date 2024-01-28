from package.constants import *
from package.llm import LLM
from package.utils import *
from package.trajectories import *
import package.skills as SKILLS
import package.reward_functions as REWARD_FUNCTIONS
from package.enums import *

import gymnasium
from typing import Callable, Dict, List, Tuple, Any, Union


class Agent:
    def __init__(self, query_source: str, model_source: str):
        self.world_model = None
        self.skills = []
        self.rewards_and_weights = []

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
    

    def speak(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")
    

    def listen(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")


class Principal(Agent):
    allowable_modes = ["demo_language", "demo_trajectory", "adapt_language", "adapt_trajectory"]

    
    def __init__(self, query_source: str, model_source: str, name: str = None):
        self.name = name if name else "Principal"
        super().__init__(query_source, model_source)
    
    
    def speak(self, mode: str, skills: List[str] = None, world_model: gymnasium.Env = None) -> Any:
        assert mode in Principal.allowable_modes, "Invalid mode"
        if mode.starstwith("demo"):
            assert (not skills and not world_model), "Demo mode does not require any input"
        elif mode.startswith("adapt"):
            assert (skills and world_model), "Adapt mode requires skills and world model inputs"
        
        if mode == "demo_language":
            return self.generate_trajectory_description()
        if mode == "demo_trajectory":
            trajectories = []
            for _ in range(10):
                trajectories.append(self.generate_trajectory())
            return trajectories
        if mode == "adapt_language":
            return self.generate_trajectory_description(skills, world_model)
        if mode == "adapt_trajectory":
            return self.generate_trajectory(skills, world_model)


    def listen(self, trajectory: Trajectory) -> bool:
        total_reward = 0
        for trans in trajectory:
            total_reward += self.get_reward(trans.obs)
        return total_reward > 0


    def generate_trajectory(self,
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


    def generate_trajectory_description(self, skills: List[str], world_model: gymnasium.Env) -> str:
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


    def listen(self, utterance: str = None, trajectories: List[Trajectory] = None) -> Trajectory:
        if utterance:
            return self.follow_instruction(utterance)
        elif trajectories:
            return self.follow_trajectories(trajectories)
    

    def follow_instruction(self, utterance: str) -> Trajectory:
        pass


    def follow_trajectories(self, trajectories: List[Trajectory]) -> Trajectory:
        pass


if __name__ == "__main__":
    la = Attendant("Taniqua", "openai", "gpt-3.5-turbo")
