from typing import Tuple

from minigrid.core.world_object import Door, WorldObj
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
)

from mindgrid.infrastructure.env_constants import (  # must keep import to pass object creation assertions
    COLOR_TO_IDX,
    COLORS,
    OBJECT_TO_IDX,
)


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


class DoorWithDirection(Door):

    def __init__(
        self,
        color: str,
        dir_vec: Tuple[int, int],
        is_open: bool = False,
        is_locked: bool = False,
    ):
        super().__init__(color, is_open=is_open, is_locked=is_locked)
        self.dir_vec = dir_vec


class Passage(WorldObj):

    def __init__(self, dir_vec: Tuple[int, int]):
        super().__init__("passage", "grey")
        self.dir_vec = dir_vec

    def can_overlap(self):
        return True

    def render(self, img):

        points = [[0.3, 0.2], [0.3, 0.8], [0.8, 0.5]]
        if self.dir_vec[1] == 0:
            if self.dir_vec == (-1, 0):
                for p in points:
                    p[0] = 1 - p[0]
        else:
            for i, p in enumerate(points):
                points[i] = [p[1], p[0]]
            if self.dir_vec == (0, -1):
                for p in points:
                    p[1] = 1 - p[1]

        fill_coords(img, point_in_triangle(*points), COLORS["grey"])

    def encode(self):
        return (
            OBJECT_TO_IDX[self.type],
            COLOR_TO_IDX[self.color],
            self.dir_vec[0] * 2 + self.dir_vec[1],
        )


class SafeLava(WorldObj):
    def __init__(self):
        super().__init__("safe_lava", "grey")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (47, 79, 79)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class SafeLava(WorldObj):
    def __init__(self):
        super().__init__("safe_lava", "grey")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (47, 79, 79)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Bridge(WorldObj):

    def __init__(self, dir_vec: Tuple[int, int], is_intact: bool = True):
        super().__init__("bridge", "brown")
        self.is_intact = is_intact
        self.dir_vec = dir_vec

    def can_overlap(self):
        return True

    def render(self, img):
        if self.is_intact:
            c = (165, 42, 42)
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)
            cc = (146, 102, 57)
            fill_coords(img, point_in_line(0.33, 0.1, 0.33, 0.9, r=0.03), cc)
            fill_coords(img, point_in_line(0.67, 0.1, 0.67, 0.9, r=0.03), cc)
        else:
            c = (165, 42, 42)
            fill_coords(img, point_in_rect(0, 1, 0, 1), c)
            cc = (255, 128, 0)
            fill_coords(img, point_in_line(0.0, 0.0, 1.0, 1.0, r=0.03), cc)
            fill_coords(img, point_in_line(0.0, 1.0, 1.0, 0.0, r=0.03), cc)

    def toggle(self, env, pos):
        if self.is_intact:
            return True
        if isinstance(env.carrying, Hammer):
            self.is_intact = True
            return True
        return False

    def encode(self):
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.is_intact)


class Hammer(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("hammer", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.15, 0.95, 0.4, 0.6), COLORS["brown"])
        fill_coords(img, point_in_rect(0.2, 0.5, 0.25, 0.75), COLORS["grey"])


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
