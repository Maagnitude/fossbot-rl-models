This repo contains the code used, to develop and evaluate **RL models** for the **FOSSBot** open source educational robot, for my **Bachelor Thesis**.

## Running TensorBoard server (on port 6006)
```
tensorboard --logdir=logs
```

## Running PPO model
```
python train_ppo.py
```

## PPO Exponential Moving Average Chart at 2000 updates
![PPO Chart - Grid 15x15 - 2000 updates](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/147d896f-27f0-4aef-ace7-2a5fb427d2ae)

## Running DQN model
```
python train_dqn.py
```

## DQN Exponential Moving Average Chart at 2500 updates
![DQN Chart - Grid 15x15 - 2500 updates](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/abff5971-b81e-4cad-a53d-41ddf3037a2a)

## DQN path to goal after 2500 updates
![dqn_gridworld_15x15](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/6e5dc24c-1ca7-4234-a510-6158d6e4e0ee)

## Running TRPO model
```
python train_trpo.py
```

## TRPO Exponential Moving Average Chart at 2500 updates
![TRPO Chart - Grid 15x15 - 2500 updates](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/2383ae67-9102-495f-b19c-0d2576a047b8)

Tool used for the charts: Weights and Biases -> https://wandb.ai
