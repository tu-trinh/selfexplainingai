from package.trajectories import Trajectory
from package.message import Message
from package.enums import MessageType, Task, Level
from package.search import Search
from package.llm import LLM
import package.reward_functions as REWARD_FUNCTIONS
from package.skills import _find_path, _check_clear_pos
from package.infrastructure.basic_utils import debug, get_adjacent_cells
from package.infrastructure.env_utils import get_obs_desc
from package.infrastructure.env_constants import IDX_TO_ACTION
from package.infrastructure.obj_constants import OBJ_NAME_MAPPING, NAME_OBJ_MAPPING
from package.task_tree import TaskNode
from package.envs.modifications import Bridge

from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box, WorldObj
from minigrid.core.grid import Grid
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
        # return self.skills
        skill_descs = []
        skill_subset = self._get_subset_of_skills()
        for skill in skill_subset:
            debug("Current skill:", skill)
            setup_actions, actions = self._retrieve_actions_from_skill_func(skill)
            obs_act_seq = self._generate_obs_act_sequence(actions, setup_actions)
            skill_desc = self.llm.get_skill_description(obs_act_seq, self.skills)
            debug("LLM called it:", skill_desc)
            skill_descs.append(skill_desc)
            debug()
        return skill_descs


    def _find_optimal_policy(self) -> List[int]:
        # FIXME: must do all BOSS levels!
        # def goal_check(env: MiniGridEnv):
            # return self.calculate_reward(env)
        # search_problem = Search("bfs", self.world_model, goal_check, "e")
        # actions = search_problem.search()

        # START ANEW!!! #
        all_actions = []
        world_model_copy = copy.deepcopy(self.world_model)
        world_model_copy.reset()

        if self.world_model.is_single_target:
            if self.world_model.level in [Level.EMPTY, Level.DEATH, Level.DIST, Level.MULT_ROOMS]:
                can_overlap = type(self.world_model.target_obj) == Goal
                all_actions.extend(self._go_directly_to_obj(world_model_copy, self.world_model.target_obj_pos, can_overlap))
            elif self.world_model.level in [Level.OPEN_DOOR, Level.GO_AROUND]:
                _, door_pos = self.world_model.doors[0]
                all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                can_overlap = type(self.world_model.target_obj) == Goal
                all_actions.extend(self._go_directly_to_obj(world_model_copy, self.world_model.target_obj_pos, can_overlap))
            elif self.world_model.level == Level.BLOCKED_DOOR:
                _, door_pos = self.world_model.doors[0]
                all_actions.extend(self._pickup_blocking_obj(world_model_copy, self.world_model.blocker_obj.init_pos))
                all_actions.extend(self._putdown_blocking_obj(world_model_copy, door_pos))
                all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                can_overlap = type(self.world_model.target_obj) == Goal
                all_actions.extend(self._go_directly_to_obj(world_model_copy, self.world_model.target_obj_pos, can_overlap))
            elif self.world_model.level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY, Level.ROOM_DOOR_KEY]:
                door, door_pos = self.world_model.doors[0]
                if self.world_model.level == Level.ROOM_DOOR_KEY:
                    if self.world_model.agent_starts_outside:
                        clear_door, blocker_obj = _check_clear_pos(world_model_copy.agent_start_pos, door_pos, world_model_copy.grid)
                        if (not clear_door) and ((type(blocker_obj) != Key) or (type(blocker_obj) == Key and (not door.is_locked or blocker_obj.color != door.color))):  # if an object other than the key we need happens to be right in front of the door
                            all_actions.extend(self._pickup_blocking_obj(world_model_copy, blocker_obj.init_pos))
                            all_actions.extend(self._putdown_blocking_obj(world_model_copy, door_pos))
                if self.world_model.level == Level.HIDDEN_KEY:
                    all_actions.extend(self._open_box_and_get_key(world_model_copy))
                elif door.is_locked and (self.world_model.level != Level.ROOM_DOOR_KEY or self.world_model.agent_starts_outside):
                    all_actions.extend(self._get_key_lying_outside(world_model_copy, door))
                # open the door (while holding the key if needed) (and if agent is outside of the room)
                if self.world_model.level != Level.ROOM_DOOR_KEY or self.world_model.agent_starts_outside:
                    all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                if self.world_model.task == Task.PICKUP:
                    # must put down key that was used to unlock the door, if holding it
                    if world_model_copy.carrying is not None:
                        all_actions.extend(self._put_down_key(world_model_copy, (self.world_model.target_obj_pos, door_pos)))
                can_overlap = type(self.world_model.target_obj) == Goal
                all_actions.extend(self._go_directly_to_obj(world_model_copy, self.world_model.target_obj_pos, can_overlap))
            elif self.world_model.level == Level.TREASURE_ISLAND:
                if self.world_model.agent_starts_outside:
                    bridge_pos = None
                    for obj, pos in self.world_model.objs:
                        if type(obj) == Bridge:
                            bridge_pos = pos
                            break
                    if bridge_pos:  # must go to bridge first before object
                        clear_bridge, blocker_obj = _check_clear_pos(world_model_copy.agent_start_pos, bridge_pos, world_model_copy.grid, is_bridge = True)
                        if not clear_bridge:  # if an object happens to be right in front of the bridge
                            all_actions.extend(self._pickup_blocking_obj(world_model_copy, blocker_obj.init_pos))
                            all_actions.extend(self._putdown_blocking_obj(world_model_copy, bridge_pos))
                        all_actions.extend(self._go_directly_to_obj(world_model_copy, bridge_pos, True))
                can_overlap = type(self.world_model.target_obj) == Goal
                all_actions.extend(self._go_directly_to_obj(world_model_copy, self.world_model.target_obj_pos, can_overlap))
            if self.world_model.task == Task.PICKUP:
                all_actions += [3]
        
        elif self.world_model.task == Task.PUT:
            first_obj_pos = self.world_model.target_objs_pos[0]
            second_obj_pos = self.world_model.target_objs_pos[1]
            put_first_next_to_second, valid_putting_place = self._determine_putting_order(first_obj_pos, second_obj_pos)
            if self.world_model.level in [Level.EMPTY, Level.DEATH, Level.DIST]:
                all_actions.extend(self._go_directly_to_obj(world_model_copy, first_obj_pos if put_first_next_to_second else second_obj_pos, False) + [3])
                all_actions.extend(self._go_directly_to_obj(world_model_copy, valid_putting_place, False) + [4])
            elif self.world_model.level in [Level.OPEN_DOOR, Level.GO_AROUND]:
                all_actions.extend(self._go_directly_to_obj(world_model_copy, first_obj_pos if put_first_next_to_second else second_obj_pos, False) + [3])
                _, door_pos = self.world_model.doors[0]
                all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                all_actions.extend(self._go_directly_to_obj(world_model_copy, valid_putting_place, False) + [4])
            elif self.world_model.level == Level.BLOCKED_DOOR:
                _, door_pos = self.world_model.doors[0]
                all_actions.extend(self._pickup_blocking_obj(world_model_copy, self.world_model.blocker_obj.init_pos))
                all_actions.extend(self._putdown_blocking_obj(world_model_copy, door_pos))
                all_actions.extend(self._go_directly_to_obj(world_model_copy, first_obj_pos if put_first_next_to_second else second_obj_pos, False) + [3])
                all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                all_actions.extend(self._go_directly_to_obj(world_model_copy, valid_putting_place, False) + [4])
            elif self.world_model.level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY, Level.ROOM_DOOR_KEY]:
                door, door_pos = self.world_model.doors[0]
                if self.world_model.level == Level.ROOM_DOOR_KEY:
                    clear_door, blocker_obj = _check_clear_pos(world_model_copy.agent_start_pos, door_pos, world_model_copy.grid)
                    if (not clear_door) and (type(blocker_obj) != Key):  # if an object other than the key we need happens to be right in front of the door
                        all_actions.extend(self._pickup_blocking_obj(world_model_copy, blocker_obj.init_pos))
                        all_actions.extend(self._putdown_blocking_obj(world_model_copy, door_pos))
                if self.world_model.level == Level.HIDDEN_KEY:
                    all_actions.extend(self._open_box_and_get_key(world_model_copy))
                elif door.is_locked:
                    all_actions.extend(self._get_key_lying_outside(world_model_copy, door))
                # open the door (while holding the key)
                all_actions.extend(self._open_door_and_go_through(world_model_copy, door_pos))
                # must put down key that was used to unlock the door
                all_actions.extend(self._put_down_key(world_model_copy, (first_obj_pos if put_first_next_to_second else second_obj_pos, door_pos)))
                all_actions.extend(self._go_directly_to_obj(world_model_copy, first_obj_pos if put_first_next_to_second else second_obj_pos, False) + [3])
                all_actions.extend(self._go_directly_to_obj(world_model_copy, valid_putting_place, False) + [4])
            elif self.world_model.level == Level.TREASURE_ISLAND:
                bridge_pos = None
                for obj, pos in self.world_model.objs:
                    if type(obj) == Bridge:
                        bridge_pos = pos
                        break
                if bridge_pos:  # must go to bridge first before object
                    clear_bridge, blocker_obj = _check_clear_pos(world_model_copy.agent_start_pos, bridge_pos, world_model_copy.grid, is_bridge = True)
                    if not clear_bridge:  # if an object happens to be right in front of the bridge
                        all_actions.extend(self._pickup_blocking_obj(world_model_copy, blocker_obj.init_pos))
                        all_actions.extend(self._putdown_blocking_obj(world_model_copy, bridge_pos))
                    all_actions.extend(self._go_directly_to_obj(world_model_copy, bridge_pos, True))
                all_actions.extend(self._go_directly_to_obj(world_model_copy, first_obj_pos if put_first_next_to_second else second_obj_pos, False) + [3])
                all_actions.extend(self._go_directly_to_obj(world_model_copy, valid_putting_place, False) + [4])
        
        self.policy = all_actions
        return all_actions
    

    def _go_directly_to_obj(self, wmc: MiniGridEnv, obj_pos: Tuple[int, int], can_overlap: bool) -> List[int]:
        actions = _find_path(wmc, obj_pos, "goto", can_overlap = can_overlap, forbidden_actions = [3, 4] if self.world_model.level == Level.MULT_ROOMS else [3, 4, 5])
        debug("GOING TO OBJECT (OR BRIDGE)", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _pickup_blocking_obj(self, wmc: MiniGridEnv, obj_pos: Tuple[int, int]) -> List[int]:
        actions = _find_path(wmc, obj_pos, "goto", forbidden_actions = [3, 4, 5]) + [3]
        debug("PICKING UP BLOCKING OBJECT", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _putdown_blocking_obj(self, wmc: MiniGridEnv, free_door_pos: Tuple[int, int]) -> List[int]:
        actions = _find_path(wmc, free_door_pos, "putdown", forbidden_actions = [3, 5])
        debug("PUTTING DOWN BLOCKING OBJECT", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _open_door_and_go_through(self, wmc: MiniGridEnv, door_pos: Tuple[int, int]) -> List[int]:
        actions = _find_path(wmc, door_pos, "goto", forbidden_actions = [3, 4, 5]) + [5, 2]
        debug("OPENING THE DOOR (MAYBE WITH KEY)", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _open_box_and_get_key(self, wmc: MiniGridEnv) -> List[int]:
        for obj, pos in self.world_model.objs:
            if type(obj) == Box and obj.contains is not None:
                box_pos = pos
                break
        actions = _find_path(wmc, box_pos, "goto", forbidden_actions = [3, 4, 5]) + [5, 3]
        debug("OPENING BOX TO GET KEY", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _get_key_lying_outside(self, wmc: MiniGridEnv, door: Door) -> List[int]:
        for key, pos in self.world_model.keys:
            if key.color == door.color:
                key_pos = pos
                break
        actions = _find_path(wmc, key_pos, "goto", forbidden_actions = [3, 4, 5]) + [3]
        debug("PICKING UP THE KEY", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _put_down_key(self, wmc: MiniGridEnv, obj_pos: Union[Tuple[int, int], Tuple[Tuple[int, int]]]) -> List[int]:
        actions = _find_path(wmc, obj_pos, "putdown", forbidden_actions = [3, 5])
        debug("PUTTING DOWN UNLOCKING KEY", actions)
        self.execute_actions(actions, wmc)
        return actions
    

    def _determine_putting_order(self, obj1_pos: Tuple[int, int], obj2_pos: [Tuple[int, int]]) -> bool:
        putting_places1 = get_adjacent_cells(obj1_pos)
        valid_putting_places1 = [pp for pp in putting_places1 if self.world_model.grid.get(*pp) is None]
        putting_places2 = get_adjacent_cells(obj2_pos)
        valid_putting_places2 = [pp for pp in putting_places2 if self.world_model.grid.get(*pp) is None]
        put_first_next_to_second = len(valid_putting_places2) > len(valid_putting_places1)
        return put_first_next_to_second, valid_putting_places2[0] if put_first_next_to_second else valid_putting_places1[0]

    
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
                        if obj_in_front is None:  # means agent was just navigating here to put something down
                            for k in range(i, j):
                                temp_tree_builder.append(tree_builder[k])
                        else:
                            obj_name = OBJ_NAME_MAPPING[type(obj_in_front)]
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
            # debug("Last added to tree builder")
            # debug(temp_tree_builder[-1].name)
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
    
    
    def _generate_obs_act_sequence(self, action_sequence: Union[List[int], TaskNode], setup_actions: List[int] = [], as_text: bool = True, fully_obs: bool = False) -> Union[str, List[Tuple[Grid, int]]]:
        detail_level = 3
        if isinstance(action_sequence, list):
            sequence = action_sequence
        else:
            sequence = action_sequence.children  # highest level skills, whatever those may be
        
        env_copy = copy.deepcopy(self.world_model)
        if fully_obs:
            env_copy = FullyObsWrapper(env_copy)
        obs, _ = env_copy.reset()
        for act in setup_actions:
            obs, _, _, _, _ = env_copy.step(act)
        carrying = None  # a helper parameter for fully obs text generation; not used in `get_obs_desc` if not fully obs
        
        # Finished setup actions above, now start to record actual actions
        if as_text:
            obs_act_seq = ""
        else:
            obs_act_seq = []
        idx = 0
        while idx < len(sequence):
            if as_text:
                obs_act_seq += f"Obs {idx + 1}: "
                obs_act_seq += get_obs_desc(obs, detail = detail_level if not fully_obs else 4, carrying = carrying)
                obs_act_seq += "\n"
                obs_act_seq += f"Act {idx + 1}: "
            if isinstance(action_sequence, list):
                if sequence[idx] == 3:
                    carrying = env_copy.grid.get(env_copy.front_pos[0], env_copy.front_pos[1])
                elif sequence[idx] == 4:
                    carrying = None
                if as_text:
                    obs_act_seq += IDX_TO_ACTION[sequence[idx]]
                else:
                    obs_act_seq.append((obs["image"], sequence[idx]))  # for raw grid purposes, need (obs img array, int) for dataset
                obs, _, _, _, _ = env_copy.step(sequence[idx])
            else:
                if "pick" in sequence[idx].name:
                    carrying = env_copy.grid.get(env_copy.front_pos[0], env_copy.front_pos[1])
                elif "put" in sequence[idx].name:
                    carrying = None
                if as_text:
                    obs_act_seq += sequence[idx].name
                else:
                    obs_act_seq.append((obs["image"], sequence[idx].name))  # kind of placeholder for now but this use case doesn't show up really
                sequence[idx].execute(env_copy)
                obs = env_copy.gen_obs()
            if as_text:
                obs_act_seq += "\n"
            idx += 1
        
        if as_text:
            obs_act_seq += "Final obs: "
            obs_act_seq += get_obs_desc(obs, detail = detail_level if not fully_obs else 4, carrying = carrying)
        else:
            obs_act_seq.append((obs["image"], None))
        return obs_act_seq
    

    def _retrieve_actions_from_skill_func(self, skill: str) -> List[int]:
        skill_func = self.world_model.allowable_skills[skill]
        basic_skill = True
        prefixes = ["go_to_", "pickup_", "put_down_", "open_", "unlock_", "close_"]
        for prefix in prefixes:
            if prefix in skill:
                color, target = self._retrieve_color_and_target_components_from_skill(prefix, skill)
                skill_type = prefix
                basic_skill = False
                break
        world_model_copy = copy.deepcopy(self.world_model)
        world_model_copy.reset()
        if basic_skill:  # covers primitive MiniGrid skills and `move` skills
            setup_actions = []  # no set up necessary, not even for forward or to find object to pick up or pick up object for put down, because we want to demonstrate the primitive skill
            actions = skill_func()
        else:
            # TODO: what if there are multiple objects that match color and target type? most obvious example is wall
            setup_actions = []
            if target == Door:
                search_list = self.world_model.doors
            elif target == Key:
                if self.world_model.level == Level.HIDDEN_KEY:
                    search_list = self.world_model.objs
                else:
                    search_list = self.world_model.keys + self.world_model.objs
            else:
                search_list = self.world_model.objs
            for obj, _ in search_list:
                if self.world_model.level == Level.HIDDEN_KEY and target == Key:
                    if type(obj) == Box and obj.contains is not None and obj.contains.color == color:
                        target_pos = obj.cur_pos if obj.cur_pos is not None else obj.init_pos
                        target_obj = obj
                        break
                else:
                    if type(obj) == target and obj.color == color:
                        target_pos = obj.cur_pos if obj.cur_pos is not None else obj.init_pos
                        target_obj = obj
                        break
            
            if skill_type == "pickup_":  # special case of hidden key because agent won't be able to pick up the key directly at first
                if self.world_model.level == Level.HIDDEN_KEY:
                    og_agent_dir = world_model_copy.agent_dir
                    setup_actions = _find_path(world_model_copy, target_pos, "goto", forbidden_actions = [3, 4, 5]) + [5]
                    self.execute_actions(setup_actions, world_model_copy)
                    additional = _find_path(world_model_copy, self.world_model.agent_start_pos, action_type = "goto", forbidden_actions = [3, 4, 5], can_overlap = True)
                    curr_agent_dir = world_model_copy.agent_dir
                    while curr_agent_dir > og_agent_dir:  # ex. currently facing up, needs to face right
                        additional.append(0)
                        curr_agent_dir -= 1
                    while curr_agent_dir < og_agent_dir:  # ex. currently facing down, needs to face up
                        additional.append(1)
                        curr_agent_dir += 1
                    setup_actions.extend(additional)
                    self.execute_actions(additional, world_model_copy)
            elif skill_type == "put_down_":
                og_agent_dir = world_model_copy.agent_dir
                setup_actions = _find_path(world_model_copy, target_pos, "goto", forbidden_actions = [3, 4, 5])
                setup_actions += [5, 3] if self.world_model.level == Level.HIDDEN_KEY else [3]
                self.execute_actions(setup_actions, world_model_copy)
                if len(self.world_model.doors) > 0:
                    target_pos = self.world_model.doors[0][1]  # pick any random door to not be blocked when agent puts down
                else:
                    target_pos = self.world_model.agent_start_pos  # no better default
                additional = _find_path(world_model_copy, self.world_model.agent_start_pos, action_type = "goto", forbidden_actions = [3, 4, 5], can_overlap = True)
                curr_agent_dir = world_model_copy.agent_dir
                while curr_agent_dir > og_agent_dir:  # ex. currently facing up, needs to face right
                    additional.append(0)
                    curr_agent_dir -= 1
                while curr_agent_dir < og_agent_dir:  # ex. currently facing down, needs to face up
                    additional.append(1)
                    curr_agent_dir += 1
                setup_actions.extend(additional)
                self.execute_actions(additional, world_model_copy)
            elif skill_type == "open_":  # close door if needed then put agent back
                if target == Door and target_obj.is_open:
                    og_agent_dir = world_model_copy.agent_dir
                    setup_actions = _find_path(world_model_copy, target_pos, "goto", forbidden_actions = [3, 4, 5]) + [5]
                    self.execute_actions(setup_actions, world_model_copy)
                    additional = _find_path(world_model_copy, self.world_model.agent_start_pos, action_type = "goto", forbidden_actions = [3, 4, 5], can_overlap = True)
                    curr_agent_dir = world_model_copy.agent_dir
                    while curr_agent_dir > og_agent_dir:  # ex. currently facing up, needs to face right
                        additional.append(0)
                        curr_agent_dir -= 1
                    while curr_agent_dir < og_agent_dir:  # ex. currently facing down, needs to face up
                        additional.append(1)
                        curr_agent_dir += 1
                    setup_actions.extend(additional)
                    self.execute_actions(additional, world_model_copy)
                else:
                    setup_actions = []
            elif skill_type == "unlock_":  # door SHOULD always be locked at the start, if unlock is a valid skill
                pass
            elif skill_type == "close_":  # open door if needed (unlock it if further needed!) then put agent back
                if target_obj.is_open:
                    setup_actions = []
                else:
                    og_agent_dir = world_model_copy.agent_dir
                    if target_obj.is_locked:
                        for key, pos in self.world_model.keys:
                            if key.color == target.color:
                                key_pos = pos
                                break
                        setup_actions = _find_path(world_model_copy, key_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [3]
                        self.execute_actions(setup_actions, world_model_copy)
                        additional = _find_path(world_model_copy, target_pos, "goto", reset = False, forbidden_actions = [3, 4, 5]) + [5]
                        setup_actions.extend(additional)
                        self.execute_actions(additional, world_model_copy)
                        additional = _find_path(world_model_copy, key_pos, "goto", forbidden_actions = [3, 4, 5]) + [4]  # put key back where it came from
                        setup_actions.extend(additional)
                        self.execute_actions(additional, world_model_copy)
                        additional = _find_path(world_model_copy, self.world_model.agent_start_pos, action_type = "goto", forbidden_actions = [3, 4, 5], can_overlap = True)  # put agent back
                        curr_agent_dir = world_model_copy.agent_dir
                        while curr_agent_dir > og_agent_dir:  # ex. currently facing up, needs to face right
                            additional.append(0)
                            curr_agent_dir -= 1
                        while curr_agent_dir < og_agent_dir:  # ex. currently facing down, needs to face up
                            additional.append(1)
                            curr_agent_dir += 1
                        setup_actions.extend(additional)
                        self.execute_actions(additional, world_model_copy)
                    else:
                        setup_actions = _find_path(world_model_copy, target_pos, "goto", forbidden_actions = [3, 4, 5]) + [5]
                        self.execute_actions(setup_actions, world_model_copy)
                        additional = _find_path(world_model_copy, self.world_model.agent_start_pos, action_type = "goto", forbidden_actions = [3, 4, 5], can_overlap = True)  # put agent back
                        curr_agent_dir = world_model_copy.agent_dir
                        while curr_agent_dir > og_agent_dir:  # ex. currently facing up, needs to face right
                            additional.append(0)
                            curr_agent_dir -= 1
                        while curr_agent_dir < og_agent_dir:  # ex. currently facing down, needs to face up
                            additional.append(1)
                            curr_agent_dir += 1
                        setup_actions.extend(additional)
                        self.execute_actions(additional, world_model_copy)
            actions = skill_func(world_model_copy, target_pos)
        return setup_actions, actions
    

    def _retrieve_color_and_target_components_from_skill(self, prefix: str, skill: str) -> Tuple[str, WorldObj]:
        components = skill.replace(prefix, "").split("_")
        if "wall" in components:
            color, target = "grey", "wall"
        elif "lava" in components:
            color, target = "red", "lava"
        elif "bridge" in components:
            color, target = "brown", "bridge"
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
