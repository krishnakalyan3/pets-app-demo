#!/usr/bin/env python3

import streamlit as st
import lightning as L
from lightning_app.frontend.stream_lit import StreamlitFrontend
from PIL import Image
from component.streamlit_component import streamlit_app
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms as T
import timm
import wandb



class LitStreamlit(L.LightningFlow):
    def configure_layout(self):
        return StreamlitFrontend(render_fn=streamlit_app)

class RootFlow(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.lit_streamlit = LitStreamlit()

    def run(self):
        self.lit_streamlit.run()

    def configure_layout(self):
        tab1 = {"name": "Streamlit", "content": self.lit_streamlit}
        return [tab1]

app = L.LightningApp(RootFlow())