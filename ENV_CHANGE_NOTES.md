## Custom Env Changes

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

#### custom_ant.xml - *gym/envs/mujoco/assets*
* new xml test model
