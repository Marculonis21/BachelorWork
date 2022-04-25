## Custom Env Changes

#### ant_v3.py - *gym/envs/mujoco*
* `healthy_z_range` - (0.2,1.0) -> (0.3,1.0) - eliminate flip over cases
* added `is_flipped` property 
* added `is_healthy` and `is_flipped` properties to the info output for checks

#### ant.xml - *gym/envs/mujoco/assets*
* added new camera to *ant.xml* - enables better tracking
