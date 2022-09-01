#!/usr/bin/env python3

from lit_jupyter import JupyterLab
import lightning as L
import os

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.jupyter_work = JupyterLab(kernel=os.getenv("LIGHTNING_JUPYTER_LAB_KERNEL", "python"), cloud_compute=L.CloudCompute(os.getenv("LIGHTNING_JUPYTER_LAB_COMPUTE", "gpu"), shm_size=4096))

    def run(self):
        self.jupyter_work.run()

    def configure_layout(self):
        tab1 = {"name": "JupyterLab", "content": self.jupyter_work}
        tab2 = {"name": "Report", "content": "https://wandb.ai/krishnakalyan/pets/reports/Doggies-Classification--VmlldzoyNTU4NDA2?accessToken=ad8hda22j5i6zywcxaaqpq28aknnop0hvoe8nlqs9ha7fe7f77fcgpd3w9ykq2gw"}

        return [tab1, tab2]

app = L.LightningApp(RootFlow())