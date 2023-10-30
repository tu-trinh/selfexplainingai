from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Door, Key, Goal, Wall
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace

class GoalEnv(MiniGridEnv):
    def __init__(self, env_id, size = 8, agent_start_pos = (1, 1), agent_start_dir = 0, max_steps = None, goals = [],
                 walls = [], doors = [], keys = [], see_through_walls = False, **kwargs):
        """
        Args:
        - env_id: string
        - size: int
        - agent_start_pos: int tuple
        - max_steps: int
        - goals: list of ((x, y), color)
        - walls: list of (x, y)
        - doors: list of ((x, y), color, is_locked)
        - keys: list of ((x, y), color)
        - see_through_walls: boolean
        """
        # Super init
        mission_space = MissionSpace(mission_func = self._gen_mission)
        if max_steps is None:
            self.max_steps = 4 * size ** 2
        else:
            self.max_steps = max_steps
        super().__init__(mission_space = mission_space, grid_size = size, max_steps = self.max_steps,
                         see_through_walls = see_through_walls, **kwargs)
        # Instance variables
        self.env_id = env_id
        self.mission = GoalEnv._gen_mission()
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goals = goals
        self.walls = walls
        self.doors = doors
        self.keys = keys
        self._gen_grid(size, size)

    @staticmethod
    def _gen_mission():
        return "go to the goal"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Set walls
        self.grid.wall_rect(0, 0, width, height)
        for wall_pos in self.walls:
            self.grid.set(wall_pos[0], wall_pos[1], Wall())
        # Place goals
        for i in range(len(self.goals)):
            goal_pos, goal_color = self.goals[i]
            assert goal_pos not in self.walls, "Goal cannot be in a wall"
            goal = Goal()
            goal.color = goal_color
            self.grid.set(goal_pos[0], goal_pos[1], goal)
        # Set keys
        for i in range(len(self.keys)):
            key_pos, key_color = self.keys[i]
            assert key_pos not in self.walls, "Key cannot be inside a wall"
            self.grid.set(key_pos[0], key_pos[1], Key(color = key_color))
        # Set doors
        for i in range(len(self.doors)):
            door_pos, door_color, is_locked = self.doors[i]
            assert door_pos in self.walls, "Door can only be for going through walls"
            self.grid.set(door_pos[0], door_pos[1], Door(color = door_color, is_locked = is_locked))
        # Place agent
        if self.agent_start_pos not in self.walls:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()