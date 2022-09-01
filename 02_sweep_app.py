#!/usr/bin/env python3

import lightning as L
from lightning_app.structures.dict import Dict
from typing import Optional
import os


class SweepWork(L.LightningWork):
    def __init__(self, cloud_compute: Optional[L.CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute,  parallel=True)

    def run(self):
        os.system(f"kaggle datasets download jessicali9530/stanford-dogs-dataset -p . --unzip")
        os.system(f"wandb agent krishnakalyan/pets/6uuuye0v")


class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.jupyter_works = Dict()

    def run(self):
        for i in range(os.getenv("SWEEP_RUN", 2)):
            work_name = f"sweep-run-{i}"
            self.jupyter_works[work_name] = SweepWork(cloud_compute=L.CloudCompute(os.getenv("LIGHTNING_JUPYTER_LAB_COMPUTE", "gpu"), shm_size=4096))
            self.jupyter_works[work_name].run()

app = L.LightningApp(RootFlow())