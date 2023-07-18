import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 6.0,
}

class CustomEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="ant.xml",
        terminate_when_unhealthy=True,
        reset_noise_scale=0.1,
        render_mode="human",
    ):
        utils.EzPickle.__init__(**locals())

        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._reset_noise_scale = reset_noise_scale

        self.step_count = 0

        # _obs = self._get_obs()
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

        MujocoEnv.__init__(self, 
                           xml_file, 
                           5, 
                           observation_space=observation_space,
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           render_mode=render_mode)

    # HARD TO DEFINE FOR DIFFERENT ROBOTS
    def is_flipped(self):
        # state = self.state_vector()
        try:
            torso_pos = self.get_body_com("torso").copy()
            bottom_pos = self.get_body_com("bottom").copy()
        except ValueError as _:
            return False

        return torso_pos[2] < bottom_pos[2] # torso is under specified bottom part

    @property
    def done(self):
        done = self.is_flipped() if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        self.step_count += 1
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        facing_direction = None
        try:
            v = self.get_body_com("front")[:2].copy() - self.get_body_com("torso")[:2].copy()
            facing_direction = v / np.linalg.norm(v) # normalized direction
        except KeyError:
            pass

        done = self.done
        observation = self._get_obs()
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "facing_direction": facing_direction,
            "is_flipped": self.is_flipped()
        }

        if self.render_mode == "human":
            self.render()

        """
        Proper observation could be sent back, but NEAT agents would be
        problematic, because this env serves many differnet custom agents -
        observation_space is not fully defined at init.
        
        NEAT agents still don't work well with this env, because of the only
        observation - int from 0 to 500 (but they don't throw exceptions).
        """
        # return observation, 0, done, False, info
        return np.array([self.step_count]), 0, done, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        observations = np.concatenate((position, velocity))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        self.step_count = 0

        # return observation
        return np.array([self.step_count])

from gymnasium.envs.registration import register
register(
     id="custom/CustomEnv-v0",
     entry_point="resources.agents.gymnasiumCustomEnv:CustomEnv",
)
