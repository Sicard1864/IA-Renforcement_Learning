# python3 -i fichier.py

import numpy as np

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):
    pi_char = ['']
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0 or i == n-1 and j == n-1:
                continue
            pi_char.append(int_to_char[pi[i,j]])
    pi_char.append('')
    return np.asarray(pi_char).reshape(n,n)

def policy_evaluation(n,pi,v,Gamma,threshhold):
  while True:
    delta = 0
    for i in range(n):
      for j in range(n):
        if i == 0 and j == 0 or i == n - 1 and j == n - 1:
          continue
        v_old = v[i,j]
        state = np.asarray([i,j])
        action = policy_one_step_look_ahead[pi[i,j]]
        next_state = state + action
        reward = -1
        if next_state[0] < 0 or next_state[0] >= n or next_state[1] < 0 or next_state[1] >= n:
          next_state = state
        v[i,j] = reward + Gamma * v[next_state[0],next_state[1]]
        delta = min(delta,abs(v_old-v[i,j]))
    if delta < threshhold:
      break
  return v

def policy_improvement(n,pi,v,Gamma):
  policy = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      if i == 0 and j == 0 or i == n-1 and j == n-1:
        continue
      q_values = []
      state = np.asarray([i,j])
      for action in policy_one_step_look_ahead.values():
        new_state = action + state
        reward = -1
        if new_state[0] < 0 or new_state[1] < 0 or new_state[0] >= n or new_state[1] >= n :
          new_state = state
        q_values.append(reward + Gamma * v[new_state[0],new_state[1]])
      policy[i,j] = np.argmax(q_values)
  if np.array_equal(pi,policy):
    return policy, True
  else:
    return policy, False

def policy_initialization(n):
  return np.random.randint(0, 4, size=(n,n))

def policy_iteration(n,Gamma,threshhold):
    pi = policy_initialization(n=n)
    v = np.zeros(shape=(n,n))
    while True:
        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)
        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)
        if pi_stable:
            break
    return pi , v


n = 4
Gamma = [0.8,0.9,1]
threshhold = 1e-4

for _gamma in Gamma:
    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)
    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)
    print()
    print(pi_char)
    print()
    print()
    print(v)
