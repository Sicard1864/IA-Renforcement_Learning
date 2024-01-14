import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    next_q_value = np.max(Q[sprime])
    Q[s][a] *= (1-alpha)
    Q[s][a] += alpha*(r + gamma * next_q_value)
    return Q


def epsilon_greedy(Q, s, epsilone):
    if (random.randint(0, 1) < epsilone):
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[s])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.8
    gamma = 0.4
    epsilon = 0.02
    n_epochs = 500
    max_itr_per_epoch = 1000
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()
        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)
            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward = ", np.mean(rewards))
    print("Training finished.\n")

    plt.plot(rewards)
    plt.title('Rewards in function of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.show()

    env.close()
