"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

# for the animation
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# for the performances
import time


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

#################################################
# 0.1 Create graph and animation
#################################################

def create_scatter(rewards, name):
    plt.figure(figsize=(10, 5))

    plt.scatter(range(1000), rewards, color='blue', alpha=0.5, label='Episode Rewards')

    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Learning Curve: Rewards per Episode')
    plt.legend()

    plt.savefig(name)

def create_epsilon_vs_reward_graph(epsilons, mean_rewards, name):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, mean_rewards, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward for Epsilon')
    plt.savefig(name)
    plt.close()

def create_agent_animation(env, agent, max_steps=200):
    frames = []
    fig, ax = plt.subplots()
    
    s, _ = env.reset()
    for _ in range(max_steps):
        frame = ax.imshow(env.render())
        frames.append([frame])
        
        a = agent.get_action(s)
        s, _, done, _, _ = env.step(a)
        
        if done:
            break
    
    ani = ArtistAnimation(fig, frames, interval=200, blit=True)
    plt.close()
    return ani

#################################################
# 0.2 - Optimization
#################################################

def optimize_agent(env, agent_class, max_epsilon, name, n_episodes=1000):
    best_epsilon = None
    best_reward = float('-inf')
    all_mean_rewards = []
    epsilons = []

    for e in np.arange(0., max_epsilon + 0.05, 0.05):
        rewards = []
        agent = agent_class(learning_rate=0.5, epsilon=e, gamma=0.99, legal_actions=list(range(env.action_space.n)))

        for _ in range(n_episodes):
            rewards.append(play_and_train(env, agent))
        
        mean_reward = np.mean(rewards[-100:])
        all_mean_rewards.append(mean_reward)
        epsilons.append(e)

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_epsilon = e
    
    create_epsilon_vs_reward_graph(epsilons, all_mean_rewards, name)
    return best_epsilon

#################################################
# 1. Play with QLearningAgent
#################################################

def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r
        
        if done:
            break
            
        s = next_s
        # END SOLUTION

    return total_reward

#################################
# 1.2 - Optimization of Qlearning
#################################
print("start of the search for the best epsilon for qlearning")
best_epsilon_q = optimize_agent(env, QLearningAgent, 0.25, "rewards/evolution_of_mean_reward_for_qlearning.png")
print("best epsilon found:", best_epsilon_q)

agent = QLearningAgent(
    learning_rate=0.5, epsilon=best_epsilon_q, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
start = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        animation = create_agent_animation(env, agent)
        animation.save(f'animations/qlearning/taxi_agent_qlearning_{i}.gif', writer='pillow')

end = time.time()
print("end training qlearning in", end-start, "secondes with mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
# TODO: créer des vidéos de l'agent en action
create_scatter(rewards, "rewards/rewards_qlearning.png")
animation = create_agent_animation(env, agent)
animation.save('animations/qlearning/taxi_agent_qlearning_after_learning.gif', writer='pillow')

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
start = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        animation = create_agent_animation(env, agent)
        animation.save(f'animations/qlearning_eps_scheduling/taxi_agent_qlearning_eps_scheduling_{i}.gif', writer='pillow')
end = time.time()

print("end training qlearning eps scheduling in", end-start, "secondes with mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
create_scatter(rewards, "rewards/rewards_qlearning_eps_scheduling.png")
animation = create_agent_animation(env, agent)
animation.save('animations/qlearning_eps_scheduling/taxi_agent_qlearning_eps_scheduling_after_learning.gif', writer='pillow')


####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
start = time.time()
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        animation = create_agent_animation(env, agent)
        animation.save(f'animations/sarsa/taxi_agent_sarsa_{i}.gif', writer='pillow')
end = time.time()

print("end training sarsa in", end-start, "secondes with mean reward", np.mean(rewards[-100:]))
assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
create_scatter(rewards, "rewards/rewards_sarsa.png")
animation = create_agent_animation(env, agent)
animation.save('animations/sarsa/taxi_agent_sarsa_after_learning.gif', writer='pillow')
