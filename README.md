


```
conda create -n wnb python=3.9 
conda activate wnb
pip install lightning
```

# Execution on Cloud

```
# Jupyter APP
lightning run app 01_jupyter_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name wnb-jupyter

# Sweep APP
lightning run app 02_sweep_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name wnb-sweep3

# Streamlit APP
lightning run app 03_serve_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--cloud --open-ui false --name wnb-sweep3
```

### Known Issues
- [] W&B report within the iFrame can you be accessed if you are not signed in.