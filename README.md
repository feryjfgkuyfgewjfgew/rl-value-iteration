# VALUE ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm

## VALUE ITERATION FUNCTION
### Name:NARESH.R
### Register Number:212223240104
### Include the value iteration function
```
desc=['FFSH','FFFH','HFFH','FHGF']
env = gym.make('FrozenLake-v1',desc=desc)
init_state = env.reset()
goal_state = 4
P = env.env.P
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:
### Mention the optimal policy
![image](https://github.com/user-attachments/assets/3a541512-2b01-4c3b-9255-ab37f3a019f5)

### optimal value function 
![image](https://github.com/user-attachments/assets/156f4a9c-efc7-41c9-ac70-bc0fae8deed2)

### success rate for the optimal policy.
![image](https://github.com/user-attachments/assets/aa328a62-bee2-4df2-9a77-4b84d6508b0b)


## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.


