import numpy as np
import gym
import matplotlib.pyplot as plt


class GA:
    def __init__(self, population_size, n_iteration, mutation_rate, env, env_obs_max, env_obs_min=None):
        self.env_name = env
        self.env = gym.make(env)
        self.action_space = self.env.action_space
        self.population_size = population_size
        self.population = np.random.randint(0, self.action_space, population_size)
        self.n_iteration = n_iteration
        self.mutation_rate = mutation_rate
        self.env_obs_max = env_obs_max
        self.env_obs_min = env_obs_max * -1 if env_obs_min is None else env_obs_min
        self.history = list()
        self.solution = self.population[0]

        assert np.all(self.env_obs_max >= self.env_obs_min)

    def get_action(self, obs, i=None, training=False):
        trimmed_obs = np.clip(obs, self.env_obs_min, self.env_obs_max) - self.env_obs_min

        space_splits = np.array(self.population_size[1:])
        space_splits_size = (self.env_obs_max - self.env_obs_min) / space_splits
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

            fitness_score[i] = total_reward

        return np.round(fitness_score, 3)

    def selection(self, fitness_score):
        unique_scores, unique_scores_index = np.unique(fitness_score, return_index=True)
        n_unique_scores = len(unique_scores)
        unique_scores_rank = unique_scores.argsort().argsort()

        if n_unique_scores == 1:
            selected = self.population[np.random.choice(10, 2, replace=False)]
        else:
            population = self.population[unique_scores_index]
            selected = population[unique_scores_rank >= max(n_unique_scores - 2, 0)]

        return selected

    def crossover(self, parents):
        if_crossover = np.random.rand(*self.population_size[1:]) < 0.5
        offsprings = np.zeros(parents.shape)

        offsprings[0] = np.where(if_crossover, parents[0], parents[1])
        offsprings[1] = np.where(np.logical_not(if_crossover), parents[0], parents[1])

        return offsprings

    def mutation(self, offsprings):
        values_for_mutation = np.random.randint(0, self.action_space, offsprings.shape)
        if_mutation = np.random.rand(*offsprings.shape) < self.mutation_rate
        offsprings = np.where(
            if_mutation,
            values_for_mutation,
            offsprings
        )

        return offsprings

    def get_next_generation(self, offsprings, fitness_score, rank):
        unique_scores = np.unique(fitness_score)
        n_unique_scores = len(unique_scores)

        if n_unique_scores == 1:
            self.population[np.random.choice(10, 2, replace=False)] = offsprings
        else:
            self.population[rank <= 2] = offsprings

    def ga(self):
        for i in range(self.n_iteration):
            generation_fitness = self.fitness()
            generation_fitness_ranking = generation_fitness.argsort().argsort() + 1
            if i > 0:
                if np.average(generation_fitness) > np.max(self.history):
                    self.solution = self.population[generation_fitness_ranking >= self.population_size[0]][0, ...]

            self.history.append(np.average(generation_fitness))

            selected_parents = self.selection(generation_fitness)
            offsprings = self.crossover(selected_parents)
            mutated_offsprings = self.mutation(offsprings)
            self.get_next_generation(mutated_offsprings, generation_fitness, generation_fitness_ranking)

            print(f"Generation: {i + 1}, fitness: {np.max(generation_fitness)}, {np.min(generation_fitness)}")

        self.env.close()

        plt.plot(
            np.arange(self.n_iteration),
            self.history
        )

        plt.ylim(min(self.history), max(self.history) + 1)
        plt.show()

    def render_solution(self):
        env = gym.make(self.env_name)

        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = self.get_action(observation)
            observation, reward, done, info = env.step(action)

        env.close()

    def save_solution(self):
        with open(self.env + ".npy", "wb") as f:
            np.save(f, self.solution)


if __name__ == "__main__":
    model = GA(
        population_size=[10, 4, 4, 4, 4, 2, 2],
        n_iteration=10,
        mutation_rate=0.01,
        env='Acrobot-v1',
        env_obs_max=np.array([1, 1, 1, 1, 3, 3])
    )

    model.ga()
    model.save_solution()
    # model.render_solution()
