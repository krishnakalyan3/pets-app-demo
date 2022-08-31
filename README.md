


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
```