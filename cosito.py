import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np


class RacingAntEnv(gym.Env):
    """
    Entorno personalizado de carrera usando el robot Ant-v5 de MuJoCo.

    Objetivo:
        El robot debe avanzar hasta x = 30 metros antes de 20 segundos.

    Observación:
        [x_position, x_velocity, distance_to_goal, elapsed_time_normalized, obs_original]

    Acción:
        La misma acción continua de Ant-v5.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.goal_distance = 30.0
        self.max_time = 20.0

        self.env = gym.make("Ant-v5", render_mode=render_mode)
        self.unwrapped_env = self.env.unwrapped

        self.dt = self.unwrapped_env.dt
        self.max_steps = int(self.max_time / self.dt)

        self.elapsed_steps = 0
        self.prev_x_position = 0.0

        original_obs_space = self.env.observation_space

        extra_low = np.array(
            [
                -np.inf,  # x_position
                -np.inf,  # x_velocity
                -np.inf,  # distance_to_goal
                0.0       # elapsed_time_normalized
            ],
            dtype=np.float32
        )

        extra_high = np.array(
            [
                np.inf,   # x_position
                np.inf,   # x_velocity
                np.inf,   # distance_to_goal
                1.0       # elapsed_time_normalized
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.concatenate([extra_low, original_obs_space.low.astype(np.float32)]),
            high=np.concatenate([extra_high, original_obs_space.high.astype(np.float32)]),
            dtype=np.float32
        )

        self.action_space = self.env.action_space

    def _get_x_position(self):
        return float(self.unwrapped_env.data.qpos[0])

    def _get_x_velocity(self):
        return float(self.unwrapped_env.data.qvel[0])

    def _get_original_obs(self):
        return self.unwrapped_env._get_obs().astype(np.float32)

    def _get_obs(self):
        x_position = self._get_x_position()
        x_velocity = self._get_x_velocity()
        distance_to_goal = self.goal_distance - x_position
        elapsed_time_normalized = min(
            (self.elapsed_steps * self.dt) / self.max_time,
            1.0
        )

        original_obs = self._get_original_obs()

        custom_obs = np.array(
            [
                x_position,
                x_velocity,
                distance_to_goal,
                elapsed_time_normalized
            ],
            dtype=np.float32
        )

        return np.concatenate([custom_obs, original_obs]).astype(np.float32)

    def step(self, action):
        x_before = self._get_x_position()

        _, original_reward, original_terminated, original_truncated, info = self.env.step(action)

        self.elapsed_steps += 1

        x_after = self._get_x_position()
        x_velocity = self._get_x_velocity()

        distance_moved = x_after - x_before
        distance_to_goal = self.goal_distance - x_after
        elapsed_time = self.elapsed_steps * self.dt

        reached_goal = x_after >= self.goal_distance
        time_limit_reached = elapsed_time >= self.max_time

        terminated = bool(reached_goal or original_terminated)
        truncated = bool(time_limit_reached or original_truncated)

        reward = 0.0

        # Recompensa por avanzar
        reward += 5.0 * distance_moved

        # Recompensa por velocidad hacia delante
        reward += 0.1 * x_velocity

        # Penalización por tiempo
        reward -= 0.01

        # Penalización por alejarse o retroceder
        if distance_moved < 0:
            reward -= 1.0

        # Bonificación por alcanzar la meta
        if reached_goal:
            reward += 100.0
            reward += max(0.0, self.max_time - elapsed_time) * 5.0

        # Penalización si se acaba el tiempo sin llegar
        if time_limit_reached and not reached_goal:
            reward -= 50.0

        obs = self._get_obs()

        info["x_position"] = x_after
        info["x_velocity"] = x_velocity
        info["distance_to_goal"] = distance_to_goal
        info["elapsed_time"] = elapsed_time
        info["reached_goal"] = reached_goal
        info["original_reward"] = original_reward

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.elapsed_steps = 0
        self.prev_x_position = 0.0

        _, info = self.env.reset(seed=seed, options=options)

        self.prev_x_position = self._get_x_position()

        obs = self._get_obs()

        info["x_position"] = self._get_x_position()
        info["x_velocity"] = self._get_x_velocity()
        info["distance_to_goal"] = self.goal_distance
        info["elapsed_time"] = 0.0
        info["reached_goal"] = False

        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


try:
    register(
        id="RacingAnt-v0",
        entry_point=RacingAntEnv,
    )
except Exception:
    pass


if __name__ == "__main__":
    env = gym.make("RacingAnt-v0", render_mode="human")

    obs, info = env.reset()

    for step in range(1000):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step} | "
            f"x={info['x_position']:.2f} | "
            f"v={info['x_velocity']:.2f} | "
            f"dist={info['distance_to_goal']:.2f} | "
            f"reward={reward:.2f}"
        )

        if terminated or truncated:
            print("Episodio terminado")
            print(info)
            obs, info = env.reset()

    env.close()