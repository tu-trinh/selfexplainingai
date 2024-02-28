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
    

class Bridge(WorldObj):  # have to change the minigrid constants themselves for color, type, etc.
    # FIXME: currently placeholder for bridge is a purple box
    """
    Wooden bridge to cross lava or other dangers
    """
    def __init__(self):
        super().__init__("box", "purple")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (87, 51, 15)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(3):
            xlo = 0.3 + 0.2 * i
            xhi = 0.4 + 0.2 * i
            # fill_coords(img, point_in_line(xlo, 0.1, xhi, 0.3, xhi, r = 0.03), c)
            # fill_coords(img, point_in_line(xlo, 0.3, xhi, 0.5, xhi, r = 0.03), c)
            # fill_coords(img, point_in_line(xlo, 0.5, xhi, 0.7, xhi, r = 0.03), c)
            # fill_coords(img, point_in_line(xlo, 0.7, xhi, 0.9, xhi, r = 0.03), c)


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
