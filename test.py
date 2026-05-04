# ============================================================
#  ENTRENAMIENTO DE CARRERA DE 100 METROS CON MUJOCO
#  Compatible con Windows y múltiples procesos
# ============================================================

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import imageio
import os
import torch

# --- Definición del entorno (fuera del bloque main) ---
class RacingAntEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.goal_distance = 100.0
        self.max_steps = 60
        self.elapsed_steps = 0
        self.prev_x_pos = 0.0

        self.env = gym.make('Ant-v5', render_mode=render_mode)
        self.unwrapped_env = self.env.unwrapped

        orig_obs_space = self.env.observation_space
        low = np.concatenate(([-np.inf, -np.inf], orig_obs_space.low))
        high = np.concatenate(([np.inf, np.inf], orig_obs_space.high))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self.env.action_space

    def _get_obs(self):
        x_pos = self.unwrapped_env.data.qpos[0]
        x_vel = self.unwrapped_env.data.qvel[0]
        obs_original = self.unwrapped_env._get_obs()
        return np.concatenate(([x_pos, x_vel], obs_original)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.elapsed_steps = 0
        self.prev_x_pos = 0.0
        obs_original, info = self.env.reset(seed=seed, options=options)
        return self._get_obs(), info

    def step(self, action):
        obs_original, reward_original, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1

        x_pos = self.unwrapped_env.data.qpos[0]
        x_vel = self.unwrapped_env.data.qvel[0]

        reward = 0.0
        distance_moved = x_pos - self.prev_x_pos
        reward += distance_moved * 1.0
        reward -= 0.1

        done = terminated or truncated or (self.elapsed_steps >= self.max_steps)
        if x_pos >= self.goal_distance:
            print(f"🎉 Meta alcanzada en el paso {self.elapsed_steps}!")
            reward += 100.0
            done = True

        self.prev_x_pos = x_pos
        obs = self._get_obs()
        return obs, reward, done, False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# Registrar el entorno
register(id='RacingAnt-v0', entry_point='__main__:RacingAntEnv')

# --- Función para crear entornos (fuera del bloque main) ---
def make_env():
    def _init():
        return RacingAntEnv(render_mode=None)
    return _init

# =================================================================
#  BLOQUE PRINCIPAL: Todo el código ejecutable va aquí
# =================================================================
if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    
    # Configuración del entrenamiento
    n_envs = 8
    print(f"🚀 Usando {n_envs} entornos en paralelo.")
    
    # Vector de entornos paralelos
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    # Verificar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Dispositivo de entrenamiento: {device}")
    
    # Crear modelo PPO
    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log="./logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    
    # Callback de evaluación
    eval_env = RacingAntEnv(render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    # Entrenamiento
    print("🏁 ¡Comienza el entrenamiento! (1.000.000 de pasos ambientales)")
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.save("modelo_carrera_final")
    print("✅ Entrenamiento completado.")
    
    # --- Visualización: Generar GIF ---
    print("🎬 Generando GIF del agente entrenado...")
    
    # Cargar el mejor modelo si existe
    best_model_path = "./best_model/best_model.zip"
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
    else:
        best_model = model
    
    vis_env = RacingAntEnv(render_mode='rgb_array')
    obs, _ = vis_env.reset()
    frames = []
    steps = 0
    max_steps_vis = 2000
    
    while steps < max_steps_vis:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = vis_env.step(action)
        frames.append(vis_env.render())
        steps += 1
        if terminated or truncated:
            print(f"🏁 Simulación finalizada después de {steps} pasos.")
            break
    vis_env.close()
    
    gif_path = "carrera_ant.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"✅ GIF guardado como '{gif_path}'")