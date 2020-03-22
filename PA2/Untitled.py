#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')


# In[2]:


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # Initialize thel value function
    V = np.zeros(env.nS)
    # While our value function is worse than the threshold theta
    while True:
        # Keep track of the update done in value function
        delta = 0
        # For each state, look ahead one step at each possible action and next state
        for s in range(env.spec.nS):
            v = 0
            q = 0
            # The possible next actions, policy[s]:[a,action_prob]
            for a in range(env.spec.nA): 
                # For each action, look at the possible next states, 
                for next_s in range(env.spec.nS): # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
                    # Calculate the expected value function
                    v += pi.action_prob(s, a) * TD[s, a, next_s] * (R[s, a, next_s] + env.spec.gamma * initV[next_s]) # P[s, a, s']*(R(s,a,s')+γV[s'])
                    # How much our value function changed across any states .  
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function update is below a threshold
        if delta < theta:
            break
    return np.array(V)


# In[3]:


from functools import partial
import numpy as np
from tqdm import tqdm

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

from dp import value_iteration, value_prediction
from monte_carlo import off_policy_mc_prediction_ordinary_importance_sampling as mc_ois
from monte_carlo import off_policy_mc_prediction_weighted_importance_sampling as mc_wis
from n_step_bootstrap import off_policy_n_step_sarsa as nsarsa
from n_step_bootstrap import on_policy_n_step_td as ntd


# In[4]:


class OneStateMDP(Env): # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self):
        env_spec=EnvSpec(4,3,0.9)

        super().__init__(env_spec)
        self.final_state = 1
        self.p = self.q = .25
        self.trans_mat, self.r_mat = self._build_trans_mat()

    def _build_trans_mat(self):
        trans_mat = np.zeros((4,3,4))

        trans_mat[0,1,1] = 1 
        trans_mat[1,0,3] = self.p
        trans_mat[1,0,1] = 1 - self.p
        trans_mat[3,0,0] = 1
        
        trans_mat[0,2,2] = 1
        trans_mat[2,0,0] = 1 - self.q
        trans_mat[2,0,3] = self.q

        r_mat = np.zeros((4,3,4))
        r_mat[2,0,0] = 1
        r_mat[2,0,3] = 1
        r_mat[3,0,0] = 10
        

        return trans_mat, r_mat

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = np.random.choice(self.spec.nS,p=self.trans_mat[self._state,action])
        r = self.r_mat[prev_state,action,self._state]

        if self._state == self.final_state:
            return self._state, r, True
        else:
            return self._state, r, False

class OneStateMDPWithModel(OneStateMDP,EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat

env = OneStateMDP()
env_with_model = OneStateMDPWithModel()

V_star, pi_star = value_iteration(env_with_model,np.zeros(env_with_model.spec.nS),1e-4)
print(V_star, pi_star)


# In[20]:


pi_star.action(0)


# In[ ]:





# In[ ]:




