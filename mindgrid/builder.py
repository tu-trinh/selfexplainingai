from __future__ import annotations

from typing import List

from minigrid.wrappers import FullyObsWrapper

from mindgrid.env import MindGridEnv
from mindgrid.envs.editors import Edits
from mindgrid.envs.layouts import Layouts
from mindgrid.envs.tasks import Tasks
from mindgrid.infrastructure.basic_utils import to_enum


def make_env(config):
    assert Tasks.has_value(config.task), "Task name {config.task} is invalid!"
    assert Layouts.has_value(config.layout), "Layout name {config.layout} is invalid!"
    for edit in config.edits:
        assert Edits.has_value(edit), f"Edit name {edit} is invalid!"

    task = to_enum(Tasks, config.task)
    layout = to_enum(Layouts, config.layout)
    edits = [to_enum(Edits, edit) for edit in config.edits]
    cls = create_env_class(task, layout)
    env = cls(
        config.seed,
        task,
        layout,
        edits,
        allowed_object_colors=config.allowed_object_colors,
        render_mode=config.render_mode if hasattr(config, "render_mode") else None,
    )
    #env = FullyObsWrapper(env)
    return env


def create_env_class(task: Task, layout: Layout):
    class_name = f"{task.value.__name__}_{layout.value}_Env"
    new_class = type(
        class_name,
        (
            MindGridEnv,
            task.value,
            layout.value,
            layout.value.editor,
            layout.value.solver,
        ),
        {"__init__": _custom_init},
    )
    return new_class


def _custom_init(
    self,
    env_seed: int,
    task: Tasks,
    layout: Layouts,
    edits: List[Edits],
    allowed_object_colors: List[str],
    render_mode=None,
    **kwargs,
):
    MindGridEnv.__init__(
        self,
        env_seed,
        task,
        layout,
        edits,
        allowed_object_colors=allowed_object_colors,
        render_mode=render_mode,
        **kwargs,
    )
    task.value.__init__(self)
    layout.value.__init__(self)

    self._init_task()
    self._init_layout()
    self.edit(edits)
