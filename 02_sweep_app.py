#!/usr/bin/env python3

import lightning as L
from lightning_app.structures.dict import Dict
from typing import Optional
import os


AGENT_NAME = "krishnakalyan/pets/3xhnzyb4"
NUM_SWEEPS = 3
NUM_GPU = 2

class SweepWork(L.LightningWork):
    def __init__(self, sweep_args: str, cloud_compute: Optional[L.CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute,  parallel=True)
        self.sweep_args = sweep_args

    def run(self):
        os.system(f"kaggle datasets download jessicali9530/stanford-dogs-dataset -p . --unzip")
        os.system("git config --global --add safe.directory '*'")
        os.system(self.sweep_args)

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.jupyter_works = Dict()
        self.current_workers = []

    def run(self):
        if len(self.current_workers) != NUM_GPU:
            for i in range(NUM_GPU):
                work_name = f"sweep-run-{i}"
                sweep_args = f"wandb agent --count {NUM_SWEEPS} {AGENT_NAME}"
                self.jupyter_works[work_name] = SweepWork(sweep_args, cloud_compute=L.CloudCompute(os.getenv("LIGHTNING_JUPYTER_LAB_COMPUTE", "gpu"), shm_size=4096))
                self.current_workers.append(work_name)
                self.jupyter_works[work_name].run()

app = L.LightningApp(RootFlow())
