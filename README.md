*Marek Bečvář*

# Semestral Work

Semestral work in the summer semester 2021/22 MFF-CUNI.
Work done under supervision of *RNDr. František Mrázek, CSc.* - Faculty MFF CUNI.

### WEEKLY TASKS
- [x] Try working with the mujoco environment in gym.
- [x] Genetic algorithm working in mujoco.
- [x] Play around with fitness functions.
- [x] Adjust the TAB mode camera angle (multiple camera modes).
- [ ] Try [DASK](https://dask.org/) library for multiprocessing
- [ ] Possible - evolution of algorithm variables.
- [ ] Graph dependency of variables on possible fitness (time/genom size).
- [ ] Try different genome types (interpolation, sinus functions).

Sodarace
DASK knihovna

### Custom Env Changes
#### Ant_v3
* `healthy_z_range` - (0.2,1.0) -> (0.3,1.0) - eliminate flip over cases
* added `is_flipped` property 
* added `is_healthy` and `is_flipped` properties to the info output for checks

#### ant.xml
* added new camera to *ant.xml* - enables better tracking
