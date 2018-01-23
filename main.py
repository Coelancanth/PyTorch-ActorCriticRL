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


    steps = 0
    episodes = 0
    episode_done = False




    print ' State Dimensions :- ', S_DIM
    print ' Action Dimensions :- ', A_DIM
    print ' Action Max :- ', A_MAX

    ram = buffer.MemoryBuffer(MAX_BUFFER)
    trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

    while episodes < MAX_EPISODES:
        _run_n_steps(env,steps,MAX_STEPS, ROLL_OUT_STEPS, ram, trainer, episodes, episode_done)
        trainer.optimize()




# Each iteration, each of N actors collect T timesteps of data
def _run_n_steps(env, steps, max_steps, roll_out_steps, ram, trainer, episodes, episode_done ):

    observation = env.reset()

    if(steps >= max_steps):
        state = env.reset()
        steps = 0
    states = []
    actions = []
    rewards = []

    # take n (roll out) steps, store them as np_value into buffer

    for i in range(roll_out_steps):
        env.render()
        state = np.float32(observation)
        states.append(state)

        action = trainer.get_exploitation_action(state)
        actions.append(action)

        next_observation, reward, done, info = env.step(action)
        rewards.append(reward)

        next_state = np.float32(next_observation)
        if done:
            state = env.reset()
            break

    # bound case, calculate final value
    if done:
        final_value = 0.0
        episodes += 1
        episode_done = True
    else:
        episode_done = False
        final_action = trainer.get_exploitation_action(next_state)

        # use pytorch calculate value function, then back to numpy
        # but is this value function Q function???

        action_var = Variable(torch.from_numpy(final_action))
        state_var = Variable(torch.from_numpy(next_state, final_action))


        value_var = trainer.critic.forward(state_var, action_var).detach()

        final_value = value_var.data.numpy()[0]
    discounted_rewards = discount_reward(rewards, final_value)
    steps += 1
    #
    ram.add(states, actions, discounted_rewards)


# different time, different rewards
def discount_reward(rewards, final_value):
    discounted_r = np.zeros_like(rewards)
    running_add = final_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * REWARD_GAMMA + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def run_n_steps(env, trainer):
    for epsidode in range(MAX_EPISODES):
        state = env.reset()
        states, actions, rewards = [], [], []
        episode_reward = 0
        for t in range(EPISODE_LENGTH): # in one episode
            env.render()
            action = trainer.get_exploitation_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            # calculate discounted rewards, and return them back to ppo agent as numpy array
            if(t+1) % ROLL_OUT_STEPS ==0 or t == EPISODE_LENGTH -1:
                final_action = action
                final_state = trainer.get_exploitation_action(final_action)
                action_var = Variable(torch.from_numpy(final_action))
                state_var = Variable(torch.from_numpy(final_state))
                # use pytorch calculate forward pass, then back to numpy array
                value_var = trainer.critic.forward(state_var, action_var).detach()
                final_value = value_var.data.numpy()[0]
                discount_rewards = discount_reward(rewards, final_value)
                return states, actions, discount_rewards

































































