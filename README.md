### Introduction
Lightning App is composed of Lightning Work and Lightning Flow. Start by wrapping existing scripts as Lightning Works. Lightning Works send state information to Lighting Flows. Lightning Flows send run command to Lightning Works. Distributed states and runs are serialized via event loops in Lightning Flows.

### Creating your environment
This is a demo lightning app that gradually shows us how to build lightning applications step by step. Make sure that you execute the commands below.

```
conda create -n wnb python=3.9 
conda activate wnb
pip install lightning
git clone https://github.com/krishnakalyan3/pets-app-demo
cd pets-app-demo
pip install -r requirements.txt
```

### Execution on Cloud
This is a demo lightning app that gradually shows us how to build a lightning application step by step. Serial execution is adviced as complexity varies.

```
# Jupyter Application
lightning run app 01_jupyter_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name training-app

# Sweep Application
lightning run app 02_sweep_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name sweep-app

# Streamlit Application
python -m lightning run app 03_serve_app.py --open-ui false --name streamlit-app \
--env WANDB_API_KEY=$WANDB_API_KEY  \
--cloud --open-ui false --name wnb-serve
```
### Known Issues
- Please sign in to access your W&B report.

