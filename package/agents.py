from package.trajectories import Trajectory
from package.message import Message
from package.enums import MessageType, Task
from package.search import Search
from package.llm import LLM
import package.reward_functions as REWARD_FUNCTIONS
from package.skills import _find_path, _check_clear_door
from package.infrastructure.basic_utils import debug
from package.infrastructure.env_utils import get_obs_desc
from package.infrastructure.env_constants import IDX_TO_ACTION, OBJ_NAME_MAPPING, NAME_OBJ_MAPPING
from package.task_tree import TaskNode
from package.envs.modifications import Bridge

from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box, WorldObj
from minigrid.minigrid_env import MiniGridEnv

from typing import Callable, Dict, List, Tuple, Any, Union
import time
import numpy as np
import copy


class Agent:
    def __init__(self, query_source: str, model_source: str):
        self.world_model = None
        self.skills = []
        self.rewards_and_weights = []
        self.policy = None
        self.task = None
        self.task_tree = None

        self.llm = LLM(query_source, model_source)
        self.tokens = 0
        self.additional_actions = None

        self.interactions = 0


    def set_world_model(self, env: MiniGridEnv) -> None:
        self.world_model = env
        self.task = self.world_model.mission


    def add_skill(self, skill: str) -> None:
        self.skills.append(skill)


    def add_reward_function(self, reward_function: Callable[[MiniGridEnv], float], weight: float) -> None:
        self.rewards_and_weights.append((reward_function, weight))


    def calculate_reward(self, env_state: MiniGridEnv) -> float:
        total_reward = 0
        for rf, w in self.rewards_and_weights:
            # reward_func = getattr(REWARD_FUNCTIONS, rf)
            # total_reward += w * reward_func(env_state)
            total_reward += w * rf(env_state)
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


    def execute_actions(self, actions: List[int], env_copy: MiniGridEnv = None) -> None:
        if env_copy is None:
            env = self.world_model
            self.world_model.reset()
        else:
            env = env_copy
        for action in actions:
            env.step(action)


    def speak(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")


    def listen(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by the agent itself")


    def verify(self, trajectory: Trajectory) -> bool:
        total_reward = 0
        for trans in trajectory:
            total_reward += self.calculate_reward(trans.obs)
        return total_reward > 0


    def generate_skill_descriptions(self) -> List[str]:
        return self.skills
        skill_descs = []
        skill_subset = self._get_subset_of_skills()
        for skill in skill_subset:
            debug("Current skill:", skill)
            setup_actions, actions = self._retrieve_actions_from_skill_func(skill)
            obs_act_seq = self._generate_obs_act_sequence(actions)
            skill_desc = self.llm.get_skill_description(obs_act_seq, self.skills)
            debug("LLM called it:", skill_desc)
            skill_descs.append(skill_desc)
            debug()
        return skill_descs


    def _find_optimal_policy(self) -> List[int]:
        # def goal_check(env: MiniGridEnv):
            # return self.calculate_reward(env)
        # search_problem = Search("bfs", self.world_model, goal_check, "e")
        # actions = search_problem.search()
        # START ANEW!!! #
        # FIXME: must handle multiple target objects :')
        all_actions = []
        world_model_copy = copy.deepcopy(self.world_model)
        world_model_copy.reset()
        if len(world_model_copy.doors) != 0:  # some door to handle here
            # FIXME: must handle mult rooms and boss levels where there are multiple doors
            door, door_pos = self.world_model.doors[0]
            clear_door, blocker_obj = _check_clear_door(world_model_copy.agent_start_pos, door_pos, world_model_copy.grid)
            if hasattr(self.world_model, "blocker_obj") or not clear_door:
                # go to and pick up the blocking object
                if blocker_obj is None:
                    blocker_obj = self.world_model.blocker_obj
                actions = _find_path(world_model_copy, blocker_obj.init_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [3]
                self.execute_actions(actions, world_model_copy)
                all_actions.extend(actions)
                debug("PICKING UP BLOCKING OBJECT")
                debug(actions)
                # put down the object somewhere
                actions = _find_path(world_model_copy, door_pos, "putdown", reset = False, forbidden_actions = [3, 5])
                self.execute_actions(actions, world_model_copy)
                all_actions.extend(actions)
                debug("PUTTING DOWN BLOCKING OBJECT")
                debug(actions)
            door_initially_locked = door.is_locked
            if door_initially_locked:
                if len(self.world_model.keys) == 0:  # hidden key
                    for obj, pos in self.world_model.objs:
                        if type(obj) == Box and obj.contains is not None:
                            box_pos = pos
                            break
                    # find and open the box and get the key inside
                    actions = _find_path(world_model_copy, box_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [5, 3]
                    self.execute_actions(actions, world_model_copy)
                    all_actions.extend(actions)
                    debug("OPENING BOX TO GET KEY")
                    debug(actions)
                else:  # normal key lying around
                    for key, pos in self.world_model.keys:
                        if key.color == door.color:
                            key_pos = pos
                            break
                    actions = _find_path(world_model_copy, key_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [3]
                    self.execute_actions(actions, world_model_copy)
                    all_actions.extend(actions)
                    debug("PICKING UP THE KEY")
                    debug(actions)
            # go to and open the door
            actions = _find_path(world_model_copy, door_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [5]
            self.execute_actions(actions, world_model_copy)
            all_actions.extend(actions)
            debug("OPENING THE DOOR")
            debug(actions)
            if door_initially_locked:  # must put down key that was used to unlock the door
                actions = _find_path(world_model_copy, door_pos, "putdown", reset = False, forbidden_actions = [3, 5])
                self.execute_actions(actions, world_model_copy)
                all_actions.extend(actions)
                debug("PUTTING DOWN UNLOCKING KEY")
                debug(actions)
        else:
            bridge_pos = None
            for obj, pos in self.world_model.objs:
                if type(obj) == Bridge:
                    bridge_pos = pos
                    break
            if bridge_pos:  # must go to bridge first before object
                actions = _find_path(world_model_copy, bridge_pos, "goto", can_overlap = True, reset = False, forbidden_actions = [3, 4, 5])
                self.execute_actions(actions, world_model_copy)
                all_actions.extend(actions)
                debug("GOING TO BRIDGE")
                debug(actions)
        # if all clear / finally, just go to the object
        actions = _find_path(world_model_copy, self.world_model.target_obj_pos, "goto", can_overlap = type(self.world_model.target_obj) == Goal, reset = False, forbidden_actions = [3, 4, 5])
        if self.world_model.task == Task.PICKUP:
            actions += [3]
        all_actions.extend(actions)
        debug("GOING TO OBJECT")
        debug(actions)
        self.policy = all_actions
        return all_actions
    

    def _build_task_tree(self) -> None:
        # Zeroth pass: creating task nodes for every action
        tree_builder = []
        for i in range(len(self.policy)):
            tree_builder.append(TaskNode(IDX_TO_ACTION[self.policy[i]]))
        # First pass: handling the move_dir_n_steps skills and turning backwards
        i = 0
        temp_tree_builder = []
        # debug("FULL POLICY")
        # debug(list(enumerate(self.policy)))
        while i < len(tree_builder):
            if self.policy[i] in [0, 1]:
                # debug("Caught", i)
                node = TaskNode()
                node.add_child(tree_builder[i])
                is_backwards = False
                j = i + 1
                if j < len(tree_builder):
                    # debug("j =", j)
                    if self.policy[j] == self.policy[i]:  # backwards turn
                        is_backwards = True
                        node.add_child(tree_builder[j])
                        j += 1
                    # debug("j =", j, "(if same then not backwards turn)")
                    while j < len(tree_builder) and self.policy[j] == 2:
                        node.add_child(tree_builder[j])
                        j += 1
                        # debug("j =", j)
                    # if j is still in the same spot, it's just a solo left/right/backwards action
                    if is_backwards and j == i + 2:
                        node.update_name("backward")
                        temp_tree_builder.append(node)
                    elif not is_backwards and j == i + 1:
                        temp_tree_builder.append(tree_builder[i])
                    else:
                        if is_backwards:
                            direction = "backward"
                        else:
                            direction = "left" if self.policy[i] == 0 else "right"
                        distance = j - i - 2 if is_backwards else j - i - 1
                        node.update_name(f"move_{direction}_{distance}_steps")
                        temp_tree_builder.append(node)
                    i = j
                else:
                    temp_tree_builder.append(tree_builder[i])
                    i += 1
            elif self.policy[i] == 2:
                node = TaskNode()
                node.add_child(tree_builder[i])
                j = i + 1
                while j < len(self.policy) and self.policy[j] == 2:
                    node.add_child(tree_builder[j])
                    j += 1
                if j == i + 1:  # if in same spot, solo move forward node
                    temp_tree_builder.append(tree_builder[i])
                else:
                    distance = j - i
                    node.update_name(f"move_forward_{distance}_steps")
                    temp_tree_builder.append(node)
                i = j
            else:
                temp_tree_builder.append(tree_builder[i])
                i += 1
        tree_builder = temp_tree_builder
        # debug("END OF FIRST PASS")
        # print([str(t) for t in tree_builder])
        # Second pass: handling the go-to actions
        i = 0
        temp_tree_builder = []
        env_copy = copy.deepcopy(self.world_model)
        env_copy.reset()
        # debug("FULL TREE BUILDER")
        # debug(list(enumerate([t.name for t in tree_builder])))
        while i < len(tree_builder):
            name = tree_builder[i].name
            if "move" in name or name == "forward":
                # debug("Caught", i)
                j = i + 1
                if j < len(tree_builder):
                    node = TaskNode()
                    node.add_child(tree_builder[i])
                    next_name = tree_builder[j].name
                    while "move" in next_name or "left" in next_name or "right" in next_name or "backward" in next_name:
                        # debug(j)
                        node.add_child(tree_builder[j])
                        j += 1
                        if j >= len(tree_builder):
                            break
                        else:
                            next_name = tree_builder[j].name
                    if j == i + 1:  # if j is in the same spot, it is just a solo move action
                        temp_tree_builder.append(tree_builder[i])
                        tree_builder[i].execute(env_copy)  # keep the actions moving
                    else:
                        obj_in_front = node.execute(env_copy)
                        obj_name = OBJ_NAME_MAPPING[type(obj_in_front)]
                        if obj_name in ["wall", "lava"]:
                            node.update_name(f"go_to_{obj_name}")
                        else:
                            node.update_name(f"go_to_{obj_in_front.color}_{obj_name}")
                        temp_tree_builder.append(node)
                    i = j
                else:
                    temp_tree_builder.append(tree_builder[i])
                    tree_builder[i].execute(env_copy)  # just to move the actions along
                    i += 1
            else:
                temp_tree_builder.append(tree_builder[i])
                tree_builder[i].execute(env_copy)  # just to move the actions along
                i += 1
        tree_builder = temp_tree_builder
        # debug("END OF SECOND PASS")
        # print([str(t) for t in tree_builder])
        # Third pass: higher level pick up/put down/open/close/unlock actions
        i = 0
        temp_tree_builder = []
        env_copy = copy.deepcopy(self.world_model)
        env_copy.reset()
        # debug("FULL TREE BUILDER")
        # debug(list(enumerate([t.name for t in tree_builder])))
        while i < len(tree_builder):
            name = tree_builder[i].name
            if name in ["pickup", "drop", "toggle"]:  # encountered if solo pickup/drop/toggle action right off the bat
                tree_builder[i].execute(env_copy)
                temp_tree_builder.append(tree_builder[i])
                i += 1
            else:
                j = i + 1
                if j < len(tree_builder):
                    node = TaskNode()
                    node.add_child(tree_builder[i])
                    next_name = tree_builder[j].name
                    while not ("pickup" in next_name or "drop" in next_name or "toggle" in next_name):
                        node.add_child(tree_builder[j])
                        j += 1
                        if j >= len(tree_builder):
                            break
                        else:
                            next_name = tree_builder[j].name
                    obj_in_front_before = node.execute(env_copy)
                    node.add_child(tree_builder[j])  # last child is the primitive pickup/drop/toggle action
                    obj_in_front_after = tree_builder[j].execute(env_copy)
                    if next_name == "pickup":
                        obj_type = OBJ_NAME_MAPPING[type(obj_in_front_before)]
                        obj_color = obj_in_front_before.color
                        node.update_name(f"pickup_{obj_color}_{obj_type}")
                    elif next_name == "drop":
                        obj_type = OBJ_NAME_MAPPING[type(obj_in_front_after)]
                        obj_color = obj_in_front_after.color
                        node.update_name(f"put_down_{obj_color}_{obj_type}")
                    elif next_name == "toggle":
                        obj_type = OBJ_NAME_MAPPING[type(obj_in_front_before)]
                        obj_color = obj_in_front_before.color
                        if obj_type == "door":
                            if obj_in_front_before.is_locked and not obj_in_front_after.is_locked:
                                node.update_name(f"unlock_{obj_color}_door")
                            elif obj_in_front_after.is_open:
                                node.update_name(f"open_{obj_color}_door")  # FIXME: why not 'unlock' but 'open'?
                            elif not obj_in_front_after.is_open:
                                node.update_name(f"close_{obj_color}_door")
                        else:  # this means we opened a box and box disappeared from cell (cannot close boxes)
                            node.update_name(f"open_{obj_color}_box")
                    temp_tree_builder.append(node)
                    i = j + 1
                else:
                    tree_builder[i].execute(env_copy)
                    temp_tree_builder.append(tree_builder[i])
                    i += 1
        tree_builder = temp_tree_builder
        # debug("END OF THIRD PASS")
        # print([str(t) for t in tree_builder])
        # Last pass: combine everything under the main task umbrella
        task_tree = TaskNode(self.task)
        for i in range(len(tree_builder)):
            task_tree.add_child(tree_builder[i])
        # debug("FULL TASK TREE")
        # debug(task_tree)
        self.task_tree = task_tree
    

    def generate_modified_policy(self, skill_descs: List[str]) -> str:
        if self.task_tree is None:
            if self.policy is None:
                self._find_optimal_policy()
            self._build_task_tree()
        obs_act_seq = self._generate_obs_act_sequence(self.task_tree)
        policy_desc = self.llm.get_new_plan_based_on_skills(self.task, obs_act_seq, skill_descs)
        return policy_desc


    def _get_subset_of_skills(self) -> List[str]:
        # TODO: hmmmmm
        return self.skills
    
    
    def _generate_obs_act_sequence(self, action_sequence: Union[List[int], TaskNode]) -> str:
        detail_level = 1
        if isinstance(action_sequence, list):
            sequence = action_sequence
        else:
            sequence = action_sequence.children  # highest level skills, whatever those may be
        obs_act_seq = ""
        env_copy = copy.deepcopy(self.world_model)
        obs, _ = env_copy.reset()
        idx = 0
        while idx < len(sequence):
            obs_act_seq += f"Obs {idx + 1}: "
            obs_act_seq += get_obs_desc(obs, detail = detail_level)
            obs_act_seq += "\n"
            obs_act_seq += f"Act {idx + 1}: "
            if isinstance(action_sequence, list):
                obs_act_seq += IDX_TO_ACTION[sequence[idx]]
                obs, _, _, _, _ = env_copy.step(action_sequence[idx])
            else:
                obs_act_seq += sequence[idx].name
                sequence[idx].execute(env_copy)
                obs = env_copy.gen_obs()
            obs_act_seq += "\n"
            idx += 1
        obs_act_seq += "Final obs: "
        obs_act_seq += get_obs_desc(obs, detail = detail_level)
        return obs_act_seq
    

    def _retrieve_actions_from_skill_func(self, skill: str) -> List[int]:
        skill_func = self.world_model.allowable_skills[skill]
        basic_skill = True
        prefixes = ["go_to_", "pickup_", "put_down", "open_", "unlock_", "close_"]
        for prefix in prefixes:
            if prefix in skill:
                color, target = self._retrieve_color_and_target_components_from_skill(prefix, skill)
                skill_type = prefix
                basic_skill = False
                break
        if basic_skill:  # covers primitive MiniGrid skills, `move` skills, and `put_down` skills
            setup_actions = []
            actions = skill_func()
        else:
            # TODO: what if there are multiple objects that match color and target type? most obvious example is wall
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
            if skill_type == "put_down":
                pass
            elif skill_type == "open_":
                pass
            elif skill_type == "unlock_":
                pass
            elif skill_type == "close_":
                pass
            else:
                setup_actions = []
            actions = skill_func(self.world_model, target_pos)
        return setup_actions, actions
    

    def _retrieve_color_and_target_components_from_skill(self, prefix: str, skill: str) -> Tuple[str, WorldObj]:
        components = skill.replace(prefix, "").split("_")
        if "wall" in components:
            color, target = "grey", "wall"
        elif "lava" in components:
            color, target = "red", "lava"
        else:
            color, target = components[0], components[1]
        return color, NAME_OBJ_MAPPING[target]


class Principal(Agent):
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
            return Message(MessageType.LANGUAGE_PLAN, new_plan)
        # TODO: add other listens


    def _generate_env_image(self):
        fully_obs_env = FullyObsWrapper(self.world_model)
        obs, _ = fully_obs_env.reset()
        # or RGBImgObsWrapper(...)?
        return obs["image"]


    def _generate_trajectory(self,
                            skills: List[str] = None,
                            world_model: MiniGridEnv = None) -> Trajectory:
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


    def _generate_trajectory_description(self, skills: List[str], world_model: MiniGridEnv) -> str:
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
