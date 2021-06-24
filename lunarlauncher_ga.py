import numpy as np
import gym
import matplotlib.pyplot as plt
from acrobot_ga import GA


class GA1(GA):
    def __init__(self, population_size, n_iteration, mutation_rate, env, env_obs_max, env_obs_min=None):
        super().__init__(population_size, n_iteration, mutation_rate, env, env_obs_max, env_obs_min)

    def get_action(self, obs, i=None, training=False):
        space_splits = np.array(self.population_size[1:])
        space_splits_size = (self.env_obs_max - self.env_obs_min) / space_splits

        trimmed_obs = np.where(
            obs >= 0,
            np.minimum(self.env_obs_max, obs),
            np.maximum(self.env_obs_min, obs)
        )
        trimmed_obs = trimmed_obs - self.env_obs_min

        idx = (trimmed_obs // space_splits_size)
        idx = np.minimum(space_splits - 1, idx).astype(int)

        if training:
            return self.population[i][tuple(idx)]
        return self.solution[tuple(idx)]

    def fitness(self):
        fitness_score = np.zeros(self.population_size[0], dtype=float)

        for i in range(self.population_size[0]):
            observation = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.get_action(observation, i, training=True)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                if abs(observation[0]) > 0.5:
                    total_reward -= 10

            fitness_score[i] = total_reward

        return np.round(fitness_score, 3)


# s (list): The state. Attributes:
# s[0] is the horizontal coordinate
# s[1] is the vertical coordinate
# s[2] is the horizontal speed
# s[3] is the vertical speed
# s[4] is the angle
# s[5] is the angular speed
# s[6] 1 if first leg has contact, else 0
# s[7] 1 if second leg has contact, else 0


model = GA(
    population_size=[10, 32, 32, 16, 16, 8, 8, 1, 1],
    action_space=4,
    n_iteration=1000,
    mutation_rate=0.05,
    env='LunarLander-v2',
    env_obs_max=np.array([0.5, 1.5, 1.5, 0, 1.5,  0.5, 1, 1], dtype=float),
    env_obs_min=np.array([-0.5, 0, -1.5, -1.5, -1.6, -0.5, 0, 0], dtype=float)
)
model.ga()
# model.save_solution()
# for _ in range(10):
#     model.render_solution()
