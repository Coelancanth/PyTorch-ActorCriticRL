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




    # for _ep in range(MAX_EPISODES):
    #     observation = env.reset()
    #     print 'EPISODE :- ', _ep
    #     for r in range(MAX_STEPS):
    #         env.render()
    #         state = np.float32(observation)
    #
    #         action = trainer.get_exploration_action(state)
    #         # if _ep%5 == 0:
    #         # 	# validate every 5th episode
    #         # 	action = trainer.get_exploitation_action(state)
    #         # else:
    #         # 	# get action based on observation, use exploration policy here
    #         # 	action = trainer.get_exploration_action(state)
    #
    #         new_observation, reward, done, info = env.step(action)
    #         # # dont update if this is validation
    #         # if _ep%50 == 0 or _ep>450:
    #         # 	continue
    #
    #         if done:
    #             new_state = None
    #         else:
    #             new_state = np.float32(new_observation)
    #             # push this exp in ram
    #             ram.add(state, action, reward, new_state)
    #
    #         observation = new_observation
    #
    #         # perform optimization
    #         trainer.optimize()
    #         if done:
    #             break
    #
    #     # check memory consumption and clear memory
    #     gc.collect()
    #     # process = psutil.Process(os.getpid())
    #     # print(process.memory_info().rss)
    #
    #     if _ep%100 == 0:
    #         trainer.save_models(_ep)
    #
    #
    # #print 'Completed episodes'


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
    rewards = discount_reward(rewards, final_value)
    steps += 1
    ram.add(states, actions, rewards)



def discount_reward(rewards, final_value):
    discounted_r = np.zeros_like(rewards)
    running_add = final_value
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * REWARD_GAMMA + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

































































