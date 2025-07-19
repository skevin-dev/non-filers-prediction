# non-filers-prediction

# Set up a working environment 


```

git clone https://github.com/skevin-dev/non-filers-prediction.git
cd non-filers-predictions
poetry install 

```


# Environment Variables

At the root folder, create a .env file and add the following variables 

```

- model directory to add models and feature columns used to train models
- data folder used to train the model 


```

# Train the model 

to train the model the model you can the following command on terminal 

```

poetry run python main,py 

```

# to use the model for prediction 

to use the model for prediction you can the following command on terminal 

```

poetry run uvicorn api:app --reload --port 8000

```


---------------------------------------------------------------------------------
