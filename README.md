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

## Links

- [Env (windows)]()
- [Env (Linux)]()
- [Training data]()
- [Testing data]() 

