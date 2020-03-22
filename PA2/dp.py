#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    TD = env.TD
    R = env.R
    Q = np.zeros((env.spec.nS, env.spec.nA))
    
    while True:
        delta = 0
        for state in range(env.spec.nS):
            v = initV[state]
            sumV, sumQ = 0, 0
            for action in range(env.spec.nA):
                sumQ = sum(TD[state, action, next_state] * (R[state, action, next_state] + env.spec.gamma * initV[next_state]) for next_state in range(env.spec.nS))
                Q[state, action] = sumQ
                sumV += pi.action_prob(state, action) * sumQ
            initV[state] = sumV
            
            delta = max(delta, abs(v - initV[state]))
        if delta < theta:
            break
            
    return initV, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    
    class Vi_policy(Policy):
        def __init__(self, _policy_probs: np.array):
            self.policy_probs = _policy_probs

        def action_prob(self,state:int,action:int) -> float:
            """
            input:
                state, action
            return:
                \pi(a|s)
            """        
            return self.policy_probs[state, action]

        def action(self,state:int) -> int:
            """
            input:
                state
            return:
                action
            """
            return np.argmax(self.policy_probs[state])

    TD = env.TD
    R = env.R
    
    while True:
        delta = 0
        for state in range(env.spec.nS):
            v = initV[state] 
            a_vals = np.zeros((env.spec.nA))
            for action in range(env.spec.nA):
                current_a = 0
                for next_state in range(env.spec.nS):
                    current_a += TD[state, action, next_state] * (R[state, action, next_state] + env.spec.gamma * initV[next_state])
                a_vals[action] = current_a
            initV[state] = np.max(a_vals)
            delta = max(delta, abs(v - initV[state]))
        if delta < theta:
            break
        
    # Find optimal policy after values have converged
    policy_probs = np.zeros((env.spec.nS, env.spec.nA))
    for state in range(env.spec.nS): 
        a_vals = np.zeros((env.spec.nA))
        for action in range(env.spec.nA):
            current_a = 0
            for next_state in range(env.spec.nS):
                current_a += TD[state, action, next_state] * (R[state, action, next_state] + env.spec.gamma * initV[next_state])
            a_vals[action] = current_a
        max_a_idx = np.argmax(a_vals)
        policy_probs[state, max_a_idx] = 1.0
        
    pi = Vi_policy(policy_probs)
    return initV, pi

