# STR-GODEs: Spatial–Temporal-Ridership Graph ODEs for Metro Ridership Prediction

This repository is the official implementation of STR-GODEs. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:
- python3
- numpy
- yaml
- pytorch
- torch_geometric
- torchdiff

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

### Extract dataset
```
cd data && tar xvf data.tar.gz
```

## Training

To train the model(s) in the paper, run this command:

- SHMetro
```
python ggnn_train.py --config
data/model/ggnn_sh_multigraph_rnn256_global_local_fusion_input.yaml
```

- HZMetro
```
python ggnn_train.py --config
data/model/ggnn_hz_multigraph_rnn256_global_local_fusion_input.yaml
```

## Evaluation

To evaluate my model, run:

- SHMetro
```
python ggnn_evaluation.py --config 
data/model/ggnn_sh_multigraph_rnn256_global_local_fusion_input.yaml
```
- HZMetro
```
python ggnn_evaluation.py --config 
data/model/ggnn_hz_multigraph_rnn256_global_local_fusion_input.yaml
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model STR-GODEs | metrics | 15min | 30min | 45min | 60min |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     85%         |      95%       |     85%         |      95%       |
| experiment1_HZMetro  |     RMSE         |     85%         |      95%       |     85%         |      95%       |
|                      |     MAPE         |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     85%         |      95%       |     85%         |      95%       |
| experiment1_SHMetro  |     RMSE         |     85%         |      95%       |     85%         |      95%       |
|                      |     MAPE         |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     31.68        |      32.37       |     32.06         |      30.82       |
| experiment2_HZMetro  |     RMSE         |     48.78         |      50.33       |     50.69         |      50.45       |
|                      |     MAPE         |     9.31%         |      9.18%       |     9.62%         |     10.20%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     35.17         |      35.81       |     36.21         |      35.44       |
| experiment2_SHMetro  |     RMSE         |     61.93         |      64.34       |     67.16        |      66.99       |
|                      |     MAPE         |     13.03%         |      13.20%       |     13.78%         |      14.87%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     17.70         |      18.25       |     18.19         |      18.45       |
| experiment3_HZMetro  |     RMSE         |     37.31         |      36.04       |     37.52         |      37.41       |
|                      |     MAPE         |     10.14%         |      10.75%       |     10.08%         |      10.74%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |
|                      |     MAE         |     16.11         |      15.69       |     14.96         |      15.66       |
| experiment3_SHMetro  |     RMSE         |     38.66         |      37.36       |     36.33         |      39.72       |
|                      |     MAPE         |     11.94%         |      11.96%       |     11.19%         |      11.02%       |
| ------------------ |---------------- |---------------- | -------------- |---------------- | -------------- |


>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
