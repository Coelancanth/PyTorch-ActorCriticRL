from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
CILP_PARAM = 0.2


class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, states=None, actions=None, rewards=None):
#    def __init__(self, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

        self.steps = 0
	
	self.states = states
	self.actions = actions 
	self.rewards = rewards

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """

        states = Variable(torch.from_numpy(self.states))
#	print states
        actions = Variable(torch.from_numpy(self.actions))
#	print actions
        discounted_rewards = Variable(torch.from_numpy(self.rewards))
#	print discounted_rewards
        #s2 = Variable(torch.from_numpy(s2))

        # ---------------------- optimize critic ----------------------

        self.critic_optimizer.zero_grad()
        #target_values = rewards
        values = torch.squeeze(self.critic.forward(states, actions))
        advantages = discounted_rewards - values

        critic_loss = torch.mean(torch.square(advantages))
        #critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        self.critic_optimizer.step()



        # a2 = self.target_actor.forward(s2).detach()
        # next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # # y_exp = r + gamma*Q'( s2, pi'(s2))
        # y_expected = r1 + GAMMA*next_val
        # # y_pred = Q( s1, a1)
        # y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # # compute critic loss, and update the critic
        # loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        # self.critic_optimizer.zero_grad()
        # loss_critic.backward()
        # self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        # pred_a1 = self.actor.forward(s1)
        # loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
        # self.actor_optimizer.zero_grad()
        # loss_actor.backward()
        # self.actor_optimizer.step()

        # optimize actor network
        self.actor_optimizer.zero_grad()
        values = torch.squeeze(self.target_critic.forward(states, actions))

        # TODO, use Generalized Advantage Estimator

        # action_log_probs = self.actor.forward(states)
        # action_log_probs = torch.sum(action_log_probs * actions, 1)
        # old_action_log_probs = self.target_actor(states)
        # old_action_log_probs = torch.sum(old_action_log_probs * actions, 1)
        # use exp since log, ratio = pi_new / pi_old
        action_probs = self.actor.forward(states)
        old_action_probs = self.target_actor.forward(states)
        ratio = action_probs/ old_action_probs

        # ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        # from paper, clamp works the best
        surr2 = torch.clamp(ratio, 1.0 - CILP_PARAM, 1.0 + CILP_PARAM) * advantages
        actor_loss = -torch.mean(torch.min(surr1, surr2))
        actor_loss.backward()
        self.actor_optimizer.step()





        # update target network
        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

        # if self.iter % 100 == 0:
        # 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
        # 		' Loss_critic :- ', loss_critic.data.numpy()
        # self.iter += 1



    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print 'Models saved successfully'

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print 'Models loaded succesfully'
