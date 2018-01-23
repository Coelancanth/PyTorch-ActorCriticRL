from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc

import train
import buffer

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
EPISODE_LENGTH = 200

MAX_TOTAL_REWARD = 300

ROLL_OUT_STEPS = 10

REWARD_GAMMA = 0.99

def run():
    env = gym.make('BipedalWalker-v2')
    # env = gym.make('Pendulum-v0')

    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]

    print ' State Dimensions :- ', S_DIM
    print ' Action Dimensions :- ', A_DIM
    print ' Action Max :- ', A_MAX

    #trainer = train.Trainer(S_DIM, A_DIM, A_MAX, states, actions, rewards)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX)
    

    for episodes in range(MAX_EPISODES):
	state = env.reset()		
	state = np.float32(state)
	states, actions, rewards = [], [], []

        states, actions, rewards = run_n_steps(env, trainer, state, states, actions, rewards)
	trainer.states = np.asarray(states)
	trainer.actions = np.asarray(actions)
	trainer.rewards = np.asarray(rewards)
        trainer.optimize()


def run_n_steps(env, trainer, state, states, actions, rewards):
        episode_reward = 0
        for t in range(EPISODE_LENGTH): # in one episode
            env.render()
            action = trainer.get_exploitation_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(np.float32(reward))

            state = next_state
	    state = np.float32(state)
            episode_reward += reward

            # calculate discounted rewards, and return them back to ppo agent as numpy array
            if(t+1) % ROLL_OUT_STEPS ==0 or t == EPISODE_LENGTH -1:
                final_state = state
                final_action = trainer.get_exploitation_action(final_state)
                action_var = Variable(torch.from_numpy(final_action)).unsqueeze(0)
                state_var = Variable(torch.from_numpy(final_state)).unsqueeze(0)
                # use pytorch calculate forward pass, then back to numpy array
                value_var = trainer.critic.forward(state_var, action_var).detach()
                final_value = value_var.data.numpy()[0]
                discount_rewards = discount_reward(rewards, final_value)
                return states, actions, discount_rewards




# different time, different rewards
def discount_reward(rewards, final_value):
    discounted_r = np.zeros_like(rewards)
    running_add = final_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * REWARD_GAMMA + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


if __name__ == "__main__":
	run()


