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

| Model STR-GODEs | 15min | 30min | 45min | 60min |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
|    |     85%         |      95%       |     85%         |      95%       |
| experiment1_HZMetro  |     85%         |      95%       |     85%         |      95%       |
|    |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
|    |     85%         |      95%       |     85%         |      95%       |
| experiment1_SHMetro   |     85%         |      95%       |     85%         |      95%       |
|    |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
|    |     85%         |      95%       |     85%         |      95%       |
| experiment2_HZMetro   |     85%         |      95%       |     85%         |      95%       |
|    |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
|    |     85%         |      95%       |     85%         |      95%       |
| experiment2_SHMetro   |     85%         |      95%       |     85%         |      95%       |
|    |     85%         |      95%       |     85%         |      95%       |
| ------------------ |---------------- | -------------- |---------------- | -------------- |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
