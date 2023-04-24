## Custom Env Changes - Gym Mujoco - deprecated (now using Farama Gymnasium)

#### ant_v3.py - *gym/envs/mujoco*
* `healthy_z_range` - (0.2,1.0) -> (0.3,1.0) - eliminate flip over cases
* added `is_flipped` property 
* added `is_healthy` and `is_flipped` properties to the info output for checks

#### ant.xml - *gym/envs/mujoco/assets*
* added new camera to *ant.xml* - enables better tracking
* changed dimensions of main plane

#### mujoco_env.py - *gym/envs/mujoco*
* changed MujocoEnv render function for *human mode* - able to pass `start_paused` param
    through `env.render()` function, causes environment rendering to start
    paused on init (useful for playback)

#### custom_antlike.py - *gym/envs/mujoco*
* added new custom ant-like environment for testing new model
* tried adding `is_flipped` but it's hard to define such property for possiblly different robots
* added `is_flipped` (healthy state) to custom env - robots need to have TORSO
    and specified BOTTOM part (`is_flipped` test coordinates of those two
    specific bodies)

#### custom_ant.xml - *gym/envs/mujoco/assets*
* new xml test model

#### __init__.py - *gym/envs/*
* new env must be registered in `__init__`

#### 4_stick_ant.xml - *gym/envs/mujoco/assets*
* basic test model

#### __init__.py - *gym/envs/mujoco*
* need to add import for the new robot


## mujoco_py changes
#### mjviewer.py - *mujoco_py/
* added `finish` function to the viewer base class - `finished` was referenced
    without the proper function existing, sim viewport wasn't closing properly
    without the function
