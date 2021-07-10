# STR-GODEs: Spatial–Temporal-Ridership Graph ODEs for Metro Ridership Prediction

This repository is the official implementation of STR-GODEs. 

The setting is adapted from https://github.com/HCPLab-SYSU/PVCGN.

## Requirements

To install requirements:
- python3
- numpy
- yaml
- pytorch
- torch_geometric
- torchdiff

A anaconda environment file py36.yml is provided in folder "test". You can activate it by modifying the prefix in it
```
prefix: /home/XXXXXX/.conda/envs/py36
```
and run the command：
```
conda env create -f py36.yml
```
Note that CUDA version in $PATH may be different from 10.1 in py36.yml, you may need to install the corresponding version of the library file(torch-scatter, torch-cluster, torch-sparse) in the following link: 
```
https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch_cluster-latest+cu101-cp36-cp36m-linux_x86_64.whl

pip install torch_scatter-latest+cu101-cp36-cp36m-linux_x86_64.whl

pip install torch_sparse-latest+cu101-cp36-cp36m-linux_x86_64.whl
```
and reinstall torch-geometric：
```
pip install torch-geometric
```


### Extract dataset
```
cd data && tar xvf data.tar.gz
```

## Training

To train the model(s) in the paper, run this command:

- SHMetro
```
python str_godes_train.py --config data/config/sh.yaml
```

- HZMetro
```
python str_godes_train.py --config data/config/hz.yaml
```

By setting the "irregular" variable in the configuration file(.yaml) to "true", we can conduct irregular prediction experiments.

## Evaluation

To evaluate my model, set the "save_path" variable in configuration file(.yaml) and run:

- SHMetro
```
python str_godes_evaluation.py --config data/config/sh.yaml
```
- HZMetro
```
python str_godes_evaluation.py --config data/config/hz.yaml
```

To evaluate my model in peak periods, run:

- SHMetro
```
python str_godes_evaluation_peak.py --config data/config/sh.yaml
```
- HZMetro
```
python str_godes_evaluation_peak.py --config data/config/hz.yaml
```


## Test

The pretrained models are provided in the "test" folder:
- SHMetro
```
python str_godes_evaluation.py --config test/STR_GODEs_sh.yaml
```
- HZMetro
```
python str_godes_evaluation.py --config test/STR_GODEs_hz.yaml
```


## Results

Our model achieves the following performance on :

Experiment1: conventional prediction experiment

Experiment2: conventional prediction experiment in peak periods(7:30-9:30 and 17:30-19:30) 

Experiment3: irregular prediction experiment

| Model STR-GODEs | metrics | 15min | 30min | 45min | 60min |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     22.84         |      23.24       |     23.69         |      24.25       |
| experiment1_HZMetro  |     RMSE         |     37.35         |      38.41       |     39.42         |      40.81       |
|                      |     MAPE         |     13.70%         |      13.87%       |     14.34%         |      15.37%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     23.21         |      23.63       |     24.65         |      25.56       |
| experiment1_SHMetro  |     RMSE         |     44.58         |      46.28       |     49.93         |      53.39       |
|                      |     MAPE         |     16.99%         |      17.12%       |     17.58%         |      18.25%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     31.68        |      32.37       |     32.06         |      30.82       |
| experiment2_HZMetro  |     RMSE         |     48.78         |      50.33       |     50.69         |      50.45       |
|                      |     MAPE         |     9.31%         |      9.18%       |     9.62%         |     10.20%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     35.17         |      35.81       |     36.21         |      35.44       |
| experiment2_SHMetro  |     RMSE         |     61.93         |      64.34       |     67.16        |      66.99       |
|                      |     MAPE         |     13.03%         |      13.20%       |     13.78%         |      14.87%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     17.70         |      18.25       |     18.19         |      18.45       |
| experiment3_HZMetro  |     RMSE         |     37.31         |      36.04       |     37.52         |      37.41       |
|                      |     MAPE         |     10.14%         |      10.75%       |     10.08%         |      10.74%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     16.11         |      15.69       |     14.96         |      15.66       |
| experiment3_SHMetro  |     RMSE         |     38.66         |      37.36       |     36.33         |      39.72       |
|                      |     MAPE         |     11.94%         |      11.96%       |     11.19%         |      11.02%       |
| ----------------------- |---------------- |---------------- | -------------- |---------------- | -------------- |


