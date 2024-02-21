from minigrid.core.world_object import Door, WorldObj


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

    def toggle(self, env, pos):  # FIXME: hmm logic seems sus/incomplete?
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                return True
            return False
        return False
    

class Bridge(WorldObj):  # have to change the minigrid constants themselves for color, type, etc.
    """
    Wooden bridge to cross lava or other dangers
    """
    def __init__(self):
        super().__init__("bridge", "brown")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (87, 51, 15)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
        for i in range(3):
            xlo = 0.3 + 0.2 * i
            xhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(xlo, 0.1, xhi, 0.3, xhi, r = 0.03), c)
            fill_coords(img, point_in_line(xlo, 0.3, xhi, 0.5, xhi, r = 0.03), c)
            fill_coords(img, point_in_line(xlo, 0.5, xhi, 0.7, xhi, r = 0.03), c)
            fill_coords(img, point_in_line(xlo, 0.7, xhi, 0.9, xhi, r = 0.03), c)


class FireproofShoes(WorldObj):  # have to control the lava here somehow?? also must be able to carry AND wear shoes???
    """
    Having these shoes allow the agent to safely walk on lava
    """
    def __init__(self):
        super().__init__("shoes", "black")
    
    def can_overlap(self):
        return False
    
    def render(self, img):
        c = (0, 0, 0)
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
