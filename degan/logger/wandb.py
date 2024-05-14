from datetime import datetime

import numpy as np
import pandas as pd
import wandb
from pathlib import Path


class WanDBWriter:
    def __init__(self, config):

        if config["trainer"].get("wandb_project") is None:
            raise ValueError("please specify project name for wandb")

        code_dir = Path(__file__).parent.parent.resolve().name
        wandb.init(
            project=config["trainer"].get("wandb_project"),
            name=config['trainer'].get("wandb_run_name"),
            config=config,
            settings=wandb.Settings(code_dir=code_dir)
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def _scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self._scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in
               scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_text(self, scalar_name, text):
        self.wandb.log({
            self._scalar_name(scalar_name): self.wandb.Html(text)
        }, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        self.wandb.log({self._scalar_name(table_name): self.wandb.Table(dataframe=table)},
                       step=self.step)
