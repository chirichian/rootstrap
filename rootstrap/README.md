# International Passangers

## Local Setup

## Requirements
 - Python 3.7 or higher
 - Conda. We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
 
## Install
```
checkout the
./environment.yaml

```
## OBSERVATION
``` 
This repository contains two models, an arima and an exponential smoothing.
train.py execute the two of them (arima and exponential) and using r2 score choose the best one, and save his model as an output.
```

 
### Create dataset

1. Create the basic features and target dataset. Run from inside the conda environment

```
python Rootstrap/time_series/make_dataset.py
```

This will generate two files in the `processed/` folder:
- `test_data.csv`: test data contains 12 periods (1 year)
- `train_data.csv`: train data contains the rest of completed periods

### Train model
2. All is set to train the model. Run from inside the conda environment

```
python Rootstrap/time_series/train.py
```

This will read the train and test datasets, train the models and compute metrics to choose the best 

The outputs are:
- `data/modeling/model.joblib`
- `data/processed/predictions.csv`
- `data/processed/predictions.jpg`


### Predict
3. After training a model, we can compute predictions for n periods

```
python capabilities/loyalty/predict.py
```

The outputs is a print