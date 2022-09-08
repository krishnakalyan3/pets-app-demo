#!/usr/bin/env python3

import streamlit as st
from PIL import Image
from torchvision import transforms as T
import wandb
from pl_model import ImageClassifier
from pathlib import Path
from albumentations.pytorch import ToTensorV2
import albumentations as A
from target import inverse_lbl


PROJECT = "pets"
ARTIFACT_METADATA = 'krishnakalyan/pets/model-19uabgdr:v16'


aug = A.Compose([A.Resize(225, 225), ToTensorV2(p=1.0),], p=1.0)


st.text_input(
    "Repo URL", value="https://github.com/krishnakalyan3/wandb-lighning-app"
)


doggy_image = st.file_uploader("Upload Doggy Image", type="png")
st.write("")


def prepare_model():
    run = wandb.init(project=PROJECT)
    artifact = run.use_artifact(ARTIFACT_METADATA, type='model')
    artifact_dir = artifact.download()
    model = ImageClassifier.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

if doggy_image is not None:
    #st.image(doggy_image)
    img = Image.open(doggy_image).convert('RGB')
    _transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = _transform(img)
    img = img.unsqueeze(0)

    model = prepare_model()
    prediction = model(img).argmax().item()
    st.markdown(f'<p class="big-font"> {inverse_lbl[prediction]}</p>', unsafe_allow_html=True)
