import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from settings import MODEL_TYPES, RUN_DATETIME_FORMAT, RUNS_ROOT
from functools import reduce


def ensure_model_type_directories(
    model_types: Sequence[str] = MODEL_TYPES,
    runs_root: Path = RUNS_ROOT,
) -> None:
    for model_type in model_types:
        (runs_root / model_type).mkdir(parents=True, exist_ok=True)


def create_run_directory(
    model_type: str,
    runs_root: Path = RUNS_ROOT,
    run_datetime_format: str = RUN_DATETIME_FORMAT,
) -> Path:
    model_runs_dir = runs_root / model_type.upper()
    model_runs_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_datetime = datetime.now().strftime(run_datetime_format)
        run_dir = model_runs_dir / run_datetime
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue


def save_json(data: Mapping[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2)


def save_animation_as_gif(
    images: Sequence[np.ndarray],
    filename: str | Path = "animation.gif",
    interval: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    img_display = ax.imshow(images[0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    def update(frame: int) -> list[Any]:
        img_display.set_data(images[frame])
        return [img_display]

    ani = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=True)
    ani.save(str(filename), writer="pillow", fps=1000 // interval)
    plt.close(fig)


def _pgcd_two(a: int, b: int) -> int:
    """Helper function for two numbers using your original logic."""
    while b != 0:
        a, b = b, a % b
    return a


def pgcd(*args: int) -> int:
    """Calculates the GCD for an arbitrary number of integers."""
    if not args:
        raise ValueError("At least one argument must be provided.")

    # reduce applies _pgcd_two cumulatively to the items of args
    result = reduce(_pgcd_two, args)
    return abs(result)
