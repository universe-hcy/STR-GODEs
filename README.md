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

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

- SHMetro
```
python ggnn_evaluation.py --config trained/sh.yaml
```
- HZMetro
```
python ggnn_evaluation.py --config trained/hz.yaml
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
