import sys
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.env_constants import OBJECT_TO_IDX, COLOR_TO_IDX, COLORS  # must keep import to pass object creation assertions

from minigrid.core.world_object import Door, WorldObj
from minigrid.utils.rendering import fill_coords, point_in_line, point_in_rect


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
    

class Bridge(WorldObj):
    """
    Wooden bridge to cross lava or other dangers
    """
    def __init__(self):
        super().__init__("bridge", "brown")

    def can_overlap(self):
        return True

    def render(self, img):
        c = tuple(COLORS["brown"])
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        cc = (146, 102, 57)
        fill_coords(img, point_in_line(0.33, 0.1, 0.33, 0.9, r = 0.03), cc)
        fill_coords(img, point_in_line(0.67, 0.1, 0.67, 0.9, r = 0.03), cc)


class FireproofShoes(WorldObj):  # have to control the lava here somehow?? also must be able to carry AND wear shoes???
    # FIXME: currently placeholder for fireproof shoes are a purple box
    """
    Having these shoes allow the agent to safely walk on lava
    """
    def __init__(self):
        super().__init__("box", "purple")
    
    def can_overlap(self):
        return False
    
    def render(self, img):
        c = (0, 0, 0)
        # fill_coords(img, point_in_rect(0, 1, 0, 1), c)
