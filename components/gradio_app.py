from lightning.app.components.serve import ServeGradio
from torchvision import transforms as T
import wandb
import gradio as gr
from pathlib import Path
import os
from components.target import inverse_lbl
from components.pl_model import ImageClassifier

PROJECT = "pets"
ARTIFACT_METADATA = 'krishnakalyan/pets/model-19uabgdr:v16'
class ImageServeGradio(ServeGradio):
    inputs = gr.inputs.Image(type="pil", shape=(224, 224))
    outputs = gr.outputs.Label(num_top_classes=120)

    def __init__(self, cloud_compute, parallel=True, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, parallel=parallel, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._labels = inverse_lbl

    def run(self):
        self.examples = [os.path.join(str("./sample"), f) for f in os.listdir("./sample")]
        self._transform = self._transform
        super().run()

    def predict(self, img):
        img = self._transform(img)
        img = img.unsqueeze(0)
        prediction = self.model(img).argmax().item()
        return self._labels[prediction]

    def build_model(self):
        run = wandb.init(project=PROJECT)
        artifact = run.use_artifact(ARTIFACT_METADATA, type='model')
        artifact_dir = artifact.download()
        model = ImageClassifier.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model