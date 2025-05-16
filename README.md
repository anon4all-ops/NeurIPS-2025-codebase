>📋  A template README.md for code accompanying a Machine Learning paper

# Probabilistic Robustness for Free? Revisiting Its Training with Benchmarking

This repository is the official implementation of [Probabilistic Robustness for Free? Revisiting Its Training with Benchmarking](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.1 \
    --batch_size 256 \
    --weight_decay 5e-4  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_Clean \
    --attack Clean \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.1 \
    --batch_size 256 \
    --weight_decay 5e-4  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_Clean \
    --attack Clean \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase eval \
    --beta 6.0 
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
