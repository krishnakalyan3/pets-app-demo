
### Creating your environment

```
conda create -n wnb python=3.9 
conda activate wnb
pip install lightning
```

### Execution on Cloud

```
# Jupyter APP
lightning run app 01_jupyter_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name training-app

# Sweep APP
lightning run app 02_sweep_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env KAGGLE_USERNAME=$KAGGLE_USERNAME --env KAGGLE_KEY=$KAGGLE_KEY \
--cloud --open-ui false --name sweep-app

# Streamlit APP
python -m lightning run app 03_serve_app.py --cloud --open-ui false --name streamlit-app

--env WANDB_API_KEY=$WANDB_API_KEY \
--cloud --open-ui false --name wnb-serve
```

### Known Issues
- Please sign in to access your W&B report.

### TODO
- [ ] Fix README
- [ ] Add lable dictionary
- [ ] Fix Streamlit Model serving
