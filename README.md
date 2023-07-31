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

![grid_env_im](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/ee9a553f-722e-4621-a373-ee1bae35d612)


![some_action_info](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/578f91a3-3fe1-4815-ab2a-31e8dc166ba7)


![grid_env_im2](https://github.com/Maagnitude/fossbot-rl-models/assets/89663008/5f8f7017-ea1f-46c7-b9d4-0ca5c9d3c620)
