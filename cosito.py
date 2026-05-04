import os
import time

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class RacingAntEnv(gym.Env):
    """
    Entorno personalizado de carrera usando Ant-v5 de MuJoCo.

    Objetivo:
        Avanzar 30 metros en línea recta antes de 20 segundos.

    Observación añadida:
        x_position
        y_position
        x_velocity
        y_velocity
        distance_to_goal
        lateral_deviation
        elapsed_time_normalized
        obs_original

    Acción:
        La misma acción continua de Ant-v5.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()

        self.goal_distance = 30.0
        self.max_time = 20.0
        self.max_lateral_deviation = 2.0

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
                -np.inf,  # y_position
                -np.inf,  # x_velocity
                -np.inf,  # y_velocity
                -np.inf,  # distance_to_goal
                0.0,      # lateral_deviation
                0.0       # elapsed_time_normalized
            ],
            dtype=np.float32
        )

        extra_high = np.array(
            [
                np.inf,   # x_position
                np.inf,   # y_position
                np.inf,   # x_velocity
                np.inf,   # y_velocity
                np.inf,   # distance_to_goal
                np.inf,   # lateral_deviation
                1.0       # elapsed_time_normalized
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    extra_low,
                    original_obs_space.low.astype(np.float32)
                ]
            ),
            high=np.concatenate(
                [
                    extra_high,
                    original_obs_space.high.astype(np.float32)
                ]
            ),
            dtype=np.float32
        )

        self.action_space = self.env.action_space

    def _get_x_position(self):
        return float(self.unwrapped_env.data.qpos[0])

    def _get_y_position(self):
        return float(self.unwrapped_env.data.qpos[1])

    def _get_x_velocity(self):
        return float(self.unwrapped_env.data.qvel[0])

    def _get_y_velocity(self):
        return float(self.unwrapped_env.data.qvel[1])

    def _get_original_obs(self):
        return self.unwrapped_env._get_obs().astype(np.float32)

    def _get_obs(self):
        x_position = self._get_x_position()
        y_position = self._get_y_position()
        x_velocity = self._get_x_velocity()
        y_velocity = self._get_y_velocity()

        distance_to_goal = self.goal_distance - x_position
        lateral_deviation = abs(y_position)

        elapsed_time_normalized = min(
            (self.elapsed_steps * self.dt) / self.max_time,
            1.0
        )

        custom_obs = np.array(
            [
                x_position,
                y_position,
                x_velocity,
                y_velocity,
                distance_to_goal,
                lateral_deviation,
                elapsed_time_normalized
            ],
            dtype=np.float32
        )

        original_obs = self._get_original_obs()

        return np.concatenate([custom_obs, original_obs]).astype(np.float32)

    def step(self, action):
        x_before = self._get_x_position()

        _, original_reward, original_terminated, original_truncated, info = self.env.step(action)

        self.elapsed_steps += 1

        x_after = self._get_x_position()
        y_after = self._get_y_position()
        x_velocity = self._get_x_velocity()
        y_velocity = self._get_y_velocity()

        distance_moved = x_after - x_before
        distance_to_goal = self.goal_distance - x_after
        lateral_deviation = abs(y_after)
        elapsed_time = self.elapsed_steps * self.dt

        reached_goal = x_after >= self.goal_distance
        time_limit_reached = elapsed_time >= self.max_time
        too_far_from_lane = lateral_deviation > self.max_lateral_deviation

        terminated = bool(
            reached_goal
            or original_terminated
            or too_far_from_lane
        )

        truncated = bool(
            time_limit_reached
            or original_truncated
        )

        reward = 0.0

        # Recompensa por avanzar en X
        reward += 5.0 * distance_moved

        # Recompensa por velocidad hacia delante
        reward += 0.1 * x_velocity

        # Penalización por desviarse lateralmente
        reward -= 0.5 * lateral_deviation

        # Penalización por velocidad lateral
        reward -= 0.1 * abs(y_velocity)

        # Penalización por tiempo
        reward -= 0.01

        # Penalización por retroceder
        if distance_moved < 0:
            reward -= 1.0

        # Penalización si se sale demasiado de la línea recta
        if too_far_from_lane:
            reward -= 30.0

        # Bonificación por alcanzar la meta
        if reached_goal:
            reward += 100.0
            reward += max(0.0, self.max_time - elapsed_time) * 5.0

        # Penalización si se acaba el tiempo sin llegar
        if time_limit_reached and not reached_goal:
            reward -= 50.0

        obs = self._get_obs()

        info["x_position"] = x_after
        info["y_position"] = y_after
        info["x_velocity"] = x_velocity
        info["y_velocity"] = y_velocity
        info["distance_to_goal"] = distance_to_goal
        info["lateral_deviation"] = lateral_deviation
        info["elapsed_time"] = elapsed_time
        info["reached_goal"] = reached_goal
        info["too_far_from_lane"] = too_far_from_lane
        info["original_reward"] = original_reward

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.elapsed_steps = 0

        _, info = self.env.reset(seed=seed, options=options)

        self.prev_x_position = self._get_x_position()

        obs = self._get_obs()

        info["x_position"] = self._get_x_position()
        info["y_position"] = self._get_y_position()
        info["x_velocity"] = self._get_x_velocity()
        info["y_velocity"] = self._get_y_velocity()
        info["distance_to_goal"] = self.goal_distance
        info["lateral_deviation"] = abs(self._get_y_position())
        info["elapsed_time"] = 0.0
        info["reached_goal"] = False
        info["too_far_from_lane"] = False

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

    TRAIN = True
    TEST = True

    model_path = "ppo_racing_ant"
    checkpoint_path = "ppo_racing_ant_checkpoint"

    total_timesteps = 300_000

    print("\n==============================")
    print("CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("==============================")
    print("Método de entrenamiento: PPO")
    print("Nombre completo: Proximal Policy Optimization")
    print("Política usada: MlpPolicy")
    print("Entorno base: Ant-v5 de MuJoCo")
    print("Entorno personalizado: RacingAnt-v0")
    print("Objetivo: recorrer 30 metros antes de 20 segundos")
    print("Tipo de acciones: continuas")
    print(f"Timesteps de entrenamiento: {total_timesteps}")
    print(f"Modelo final: {model_path}.zip")
    print(f"Checkpoint de emergencia: {checkpoint_path}.zip")

    training_finished_correctly = False

    if TRAIN:
        train_env = None
        model = None

        try:
            print("\n==============================")
            print("INICIO DEL ENTRENAMIENTO")
            print("==============================")

            train_env = RacingAntEnv(render_mode=None)

            check_env(train_env, warn=True)

            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
            )

            print("\nHiperparámetros principales:")
            print("learning_rate = 3e-4")
            print("n_steps = 2048")
            print("batch_size = 64")
            print("gamma = 0.99")
            print("gae_lambda = 0.95")
            print("clip_range = 0.2")
            print("ent_coef = 0.01")

            estimated_max_episodes = total_timesteps / train_env.max_steps

            print("\nEstimación aproximada:")
            print(f"Máximo de steps por episodio: {train_env.max_steps}")
            print(f"Episodios aproximados de entrenamiento: {estimated_max_episodes:.2f}")
            print("Nota: no es exacto, porque algunos episodios pueden terminar antes.")

            model.learn(total_timesteps=total_timesteps)

            model.save(model_path)
            training_finished_correctly = True

            print("\nEntrenamiento terminado correctamente.")
            print(f"Modelo guardado en: {model_path}.zip")

        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido manualmente.")
            print("Guardando checkpoint de emergencia...")

            if model is not None:
                try:
                    model.save(checkpoint_path)
                    print(f"Checkpoint guardado en: {checkpoint_path}.zip")
                except Exception as save_error:
                    print("No se pudo guardar el checkpoint.")
                    print(f"Error al guardar: {save_error}")
            else:
                print("No existe ningún modelo inicializado para guardar.")

        except Exception as error:
            print("\nEl entrenamiento ha fallado.")
            print(f"Tipo de error: {type(error).__name__}")
            print(f"Mensaje: {error}")

            if model is not None:
                try:
                    model.save(checkpoint_path)
                    print(f"Checkpoint guardado en: {checkpoint_path}.zip")
                except Exception as save_error:
                    print("No se pudo guardar el checkpoint.")
                    print(f"Error al guardar: {save_error}")
            else:
                print("No existe ningún modelo inicializado para guardar.")

        finally:
            if train_env is not None:
                train_env.close()

    if TEST:
        print("\n==============================")
        print("INICIO DE LA PRUEBA VISUAL")
        print("==============================")

        model_to_load = None

        try:
            if training_finished_correctly and os.path.exists(model_path + ".zip"):
                model_to_load = model_path
                print(f"Cargando modelo final: {model_path}.zip")

            elif os.path.exists(checkpoint_path + ".zip"):
                model_to_load = checkpoint_path
                print("El entrenamiento no terminó correctamente.")
                print(f"Cargando checkpoint: {checkpoint_path}.zip")

            elif os.path.exists(model_path + ".zip"):
                model_to_load = model_path
                print("No se ha entrenado en esta ejecución, pero existe un modelo previo.")
                print(f"Cargando modelo existente: {model_path}.zip")

            else:
                print("\nNo se ha encontrado ningún modelo para probar.")
                print("Primero debe completarse un entrenamiento o existir un checkpoint.")
                exit()

            model = PPO.load(model_to_load)

        except Exception as error:
            print("\nNo se ha podido cargar ningún modelo para la prueba.")
            print(f"Tipo de error: {type(error).__name__}")
            print(f"Mensaje: {error}")
            exit()

        test_env = None

        try:
            test_env = gym.make("RacingAnt-v0", render_mode="human")

            num_episodes = 5

            successful_episodes = 0
            failed_episodes = 0

            for episode in range(num_episodes):
                obs, info = test_env.reset()

                total_reward = 0.0
                max_x = 0.0
                max_lateral_deviation = 0.0

                print("\n==============================")
                print(f"EPISODIO DE PRUEBA {episode + 1}/{num_episodes}")
                print("==============================")
                print("Método usado por el agente: PPO entrenado")

                for step in range(test_env.unwrapped.max_steps):
                    action, _ = model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = test_env.step(action)

                    # Ralentiza el render para que se pueda ver.
                    time.sleep(0.02)

                    total_reward += reward
                    max_x = max(max_x, info["x_position"])
                    max_lateral_deviation = max(
                        max_lateral_deviation,
                        info["lateral_deviation"]
                    )

                    if step % 20 == 0:
                        print(
                            f"step={step:04d} | "
                            f"t={info['elapsed_time']:.2f}s | "
                            f"x={info['x_position']:.2f}m | "
                            f"y={info['y_position']:.2f}m | "
                            f"vx={info['x_velocity']:.2f}m/s | "
                            f"vy={info['y_velocity']:.2f}m/s | "
                            f"dist={info['distance_to_goal']:.2f}m | "
                            f"lat_dev={info['lateral_deviation']:.2f}m | "
                            f"reward={reward:.2f}"
                        )

                    if terminated or truncated:
                        break

                reached_goal = info["reached_goal"]
                too_far_from_lane = info["too_far_from_lane"]
                time_finished = info["elapsed_time"] >= test_env.unwrapped.max_time

                if reached_goal:
                    successful_episodes += 1
                    result_message = "ÉXITO: el robot ha llegado a la meta."
                else:
                    failed_episodes += 1

                    if too_far_from_lane:
                        result_message = "FALLO: el robot se ha salido de la pista."
                    elif time_finished:
                        result_message = "FALLO: el robot no ha llegado antes de 20 segundos."
                    else:
                        result_message = "FALLO: el episodio terminó por otra condición del entorno."

                print("\nResultado del episodio:")
                print(result_message)
                print("Método de entrenamiento: PPO")
                print(f"Modelo evaluado: {model_to_load}.zip")
                print(f"Steps ejecutados: {step + 1}/{test_env.unwrapped.max_steps}")
                print(f"Tiempo final: {info['elapsed_time']:.2f}s")
                print(f"Posición final X: {info['x_position']:.2f}m")
                print(f"Posición final Y: {info['y_position']:.2f}m")
                print(f"Máxima X alcanzada: {max_x:.2f}m")
                print(f"Máxima desviación lateral: {max_lateral_deviation:.2f}m")
                print(f"Distancia restante: {info['distance_to_goal']:.2f}m")
                print(f"Recompensa total: {total_reward:.2f}")

            print("\n==============================")
            print("RESUMEN FINAL DE PRUEBA")
            print("==============================")
            print("Método evaluado: PPO")
            print(f"Modelo evaluado: {model_to_load}.zip")
            print(f"Episodios totales: {num_episodes}")
            print(f"Episodios exitosos: {successful_episodes}")
            print(f"Episodios fallidos: {failed_episodes}")
            print(f"Tasa de éxito: {(successful_episodes / num_episodes) * 100:.2f}%")

        except Exception as error:
            print("\nLa prueba visual ha fallado.")
            print(f"Tipo de error: {type(error).__name__}")
            print(f"Mensaje: {error}")

        finally:
            if test_env is not None:
                input("\nPrueba terminada. Pulsa ENTER para cerrar la ventana de render...")
                test_env.close()