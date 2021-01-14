# Interaction Network

This repository is for implementing [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)



## Environment

![Env](./Image/Env.gif)



The environment is made with [Unity ML-agents](https://unity3d.com/machine-learning). There are 5 balls in the environment. They bounce when they collide with each other or collide with wall. The goal of this project is to predict position of the balls using **Interaction Network**. 

1 episode is 1000 steps. At every episode, balls are on random position at random speed. 

Each ball has different mass and size. 

The state information of each ball is as follows. 

- Position x
- Position y
- Velocity x
- Velocity y
- 1/Mass
- 1/Size

Therefore, total state vector size is 36. (6 data X (5 balls + wall))



In the dataset, there is data **X** and **Y**. Data X is input data of the Interaction network and Data Y is the next time step data of X. 

<br>

## Description of Files

- Making_Dataset.ipynb: You can make training and testing dataset using the environment
- IN_tf.ipynb: Interaction Network algorithm (Tensorflow 1.x)
- IN_torch.ipynb: Interaction Network algorithm (Torch 1.7)

<br>

## Links of files 

- [Env (windows)](https://www.dropbox.com/s/yg1j4f8k2iilub3/env.zip?dl=0) -> Unzip it inside the `env` folder
- [Env (Linux)](https://www.dropbox.com/s/6r96wkdd0y78br1/env_linux.zip?dl=0) -> Unzip it inside the `env` folder
- [Training data](https://www.dropbox.com/s/gu4kjfnhjdhfvpl/Training_dataset.mat?dl=0) -> Put training data inside `data` folder
- [Testing data](https://www.dropbox.com/s/zkxnwze42qzcl12/Testing_dataset.mat?dl=0) -> Put Testing data inside `data` folder
- [Network Variables](https://www.dropbox.com/s/964co0c52f01gj4/model.zip?dl=0) -> Put model file inside `saved_networks` folder 

