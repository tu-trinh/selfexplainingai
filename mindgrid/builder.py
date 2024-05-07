from __future__ import annotations

from typing import List

from mindgrid.env import MindGridEnv
from mindgrid.envs.editors import Edit
from mindgrid.envs.layouts import Layout
from mindgrid.envs.tasks import Task
from mindgrid.infrastructure.basic_utils import to_enum


def make_env(config):
    assert Task.has_value(config.task), "Task name {config.task} is invalid!"
    assert Layout.has_value(config.layout), "Layout name {config.layout} is invalid!"
    for edit in config.edits:
        assert Edit.has_value(edit), f"Edit name {edit} is invalid!"

    task = to_enum(Task, config.task)
    layout = to_enum(Layout, config.layout)
    edits = [to_enum(Edit, edit) for edit in config.edits]
    cls = create_env_class(task, layout)
    env = cls(
        config.seed,
        task,
        layout,
        edits,
        allowed_object_colors=config.allowed_object_colors,
        render_mode="human",
    )
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
    task: Task,
    layout: Layout,
    edits: List[Edit],
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
