from PIL import Image
from torchvision import transforms as T
import wandb
from pathlib import Path
from components.target import inverse_lbl
from components.pl_model import ImageClassifier
import streamlit as st

PROJECT = "pets"
ARTIFACT_METADATA = 'krishnakalyan/pets/model-19uabgdr:v16'
run = wandb.init(project=PROJECT)
artifact = run.use_artifact(ARTIFACT_METADATA, type='model')
artifact_dir = artifact.download()


def streamlit_app(lightning_app_state):
    model = ImageClassifier.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    st.text_input(
        "Repo URL", value="https://github.com/krishnakalyan3/wandb-lighning-app"
    )
    doggy_image = st.file_uploader("Upload Doggy Image", type="png")
    st.write("")

    if doggy_image is not None:
        img = Image.open(doggy_image).convert('RGB')
        _transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img = _transform(img)
        img = img.unsqueeze(0)
        prediction = model(img).argmax().item()
        st.markdown(f'<p class="big-font"> {inverse_lbl[prediction]}</p>', unsafe_allow_html=True)
