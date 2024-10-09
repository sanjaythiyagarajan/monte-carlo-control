### Name :SANJAY T
### Register Number : 212222110039
### Date : 09.10.2024

# MONTE CARLO CONTROL ALGORITHM

## AIM
To implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of 
goal-reaching probability and average return.

## PROBLEM STATEMENT
The task involves solving a Markov Decision Process (MDP) using Monte Carlo Control. 
The environment is likely a grid world where an agent must navigate through states to reach a goal while maximizing returns. 
The goal is to compute an optimal policy that achieves the highest probability of success (reaching the goal) and maximizes the average undiscounted return.

## MONTE CARLO CONTROL ALGORITHM

1. Initialize the policy randomly.

2. Generate episodes: Simulate episodes in the environment using the current policy.

3. Update action-value function Q(s,a): For each state-action pair encountered in the episode,
   update the expected return based on the actual rewards received during the episode.

5. Policy improvement: Update the policy greedily based on the updated action-value estimates.

6. Repeat the process until convergence.

## MONTE CARLO CONTROL FUNCTION
```python
def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=4000, max_steps=200, first_visit=True):
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    returns_count = np.zeros((nS, nA))


    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    def select_action(state, Q, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(nA)
        return np.argmax(Q[state])

    for episode in range(n_episodes):
        epsilon = epsilons[episode]
        alpha = alphas[episode]
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)

        G = 0
        visited = set()

        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = gamma * G + reward
            if (state, action) not in visited or not first_visit:
                returns_count[state, action] += 1
                Q[state, action] += alpha * (G - Q[state, action])
                visited.add((state, action))


    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)

    return Q, V, pi

```
## OUTPUT:

![image](https://github.com/user-attachments/assets/b7f2b5d9-c697-4e86-aedf-392068e84fac)

![image](https://github.com/user-attachments/assets/06e07974-b5d4-4a16-860d-76d9d369a200)



## RESULT:
Thus to implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of 
goal-reaching probability and average return is executed successfully.
