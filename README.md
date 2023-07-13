This repo contains the code used, to develop and evaluate **RL models** for the **FOSSBot** open source educational robot, for my **Bachelor Thesis**.

## Running TensorBoard server (on port 6006)
```
tensorboard --logdir=logs
```

## Running model train and save
```
python model-save.py
```

* Model is saved in models/<model_name> dir, using its timesteps as its name.
* For example, the **PPO models** for 30000 timesteps, are saved as "**models/PPO/ppo_agent_30000.zip**"
* To delete the logs, TensorBoard server must shut down.