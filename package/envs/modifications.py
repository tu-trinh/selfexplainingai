import sys
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.env_constants import OBJECT_TO_IDX, COLOR_TO_IDX, COLORS  # must keep import to pass object creation assertions

from minigrid.core.world_object import Door, Goal, WorldObj
from minigrid.utils.rendering import fill_coords, point_in_line, point_in_rect, point_in_circle


class HeavyDoor(Door):
    """
    Door that cannot be opened.
    """
    def __init__(self, color: str, is_locked: bool = False):
        super().__init__("door", color)
        self.is_locked = is_locked

    def can_overlap(self):
        return False

    def see_behind(self):
        return False

    def toggle(self, env, pos):
        return False


class SafeLava(Goal):
    def __self__(self):
        super().__init__("safe_lava", "red")

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), (86, 0, 43))


class Bridge(WorldObj):
    """
    Wooden bridge to cross lava or other dangers
    """
    def __init__(self):
        super().__init__("bridge", "brown")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (165, 42, 42)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        cc = (146, 102, 57)
        fill_coords(img, point_in_line(0.33, 0.1, 0.33, 0.9, r = 0.03), cc)
        fill_coords(img, point_in_line(0.67, 0.1, 0.67, 0.9, r = 0.03), cc)


class FireproofShoes(WorldObj):
    """
    Having these shoes allow the agent to safely walk on lava
    """
    def __init__(self):
        super().__init__("fireproof_shoes", "red")

    def can_overlap(self):
        return False

    def can_pickup(self):
        return True

    def render(self, img):
        c = (255, 0, 0)
        fill_coords(img, point_in_rect(0.4, 0.8, 0.1, 0.8), c)
        fill_coords(img, point_in_circle(0.4, 0.7, 0.15), c)
        fill_coords(img, point_in_circle(0.65, 0.7, 0.15), c)
        fill_coords(img, point_in_rect(0.35, 0.8, 0.1, 0.3), (255, 255, 255))

