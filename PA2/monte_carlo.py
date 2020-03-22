#!/usr/bin/env python
# coding: utf-8

# In[14]:


from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    c = np.zeros([env_spec.nS, env_spec.nA])
    for episode in trajs:
        g = 0
        w = 1
        
        for t in range(len(episode) - 1, -1,-1):
            if w != 0:
                st, at, rt1, st1 = episode[t]
                g = env_spec.gamma * g + rt1
                c[st, at] += 1
                initQ[st, at] += w * g
                w *= pi.action_prob(st, at)/bpi.action_prob(st, at)
            else:
                break
    

    c = np.where(c == 0, 1, c)    
    initQ = np.divide(initQ, c)
    return initQ

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    c = np.zeros([env_spec.nS, env_spec.nA])
    for episode in trajs:
        g = 0
        w = 1  
        for t in range(len(episode) - 1, -1,-1):
            if w != 0:
                st, at, rt1, st1 = episode[t]
                g = env_spec.gamma * g + rt1
                c[st, at] += w
                initQ[st, at] += w / c[st,at] * (g - initQ[st, at])
                w *= pi.action_prob(st, at)/bpi.action_prob(st, at)
            else:
                break
    return initQ

