#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    # Initialise all actions having probability zero
    action_probabilities = np.zeros(nA)
    # Choose greedy action and make its probability (1-epsilon)
    greedy_index = np.argmax(Q[state])
    action_probabilities[greedy_index]=1-epsilon
    # Make all other probabilities as (epsilon/nA)
    action_probabilities+=(epsilon/nA)
    action = np.random.choice(nA, p = action_probabilities)
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for episode_i in range(n_episodes):
        # define decaying epsilon
        if episode_i!=0:
            epsilon = 0.99*epsilon
    
        # initialize the environment 
        obs = env.reset()
        
        # get an action from policy
        action = epsilon_greedy(Q,obs,nA, epsilon = epsilon)
        # loop for each step of episode
        done = False
        while not done:
            # return a new state, reward and done
            new_state,reward,done,_=env.step(action)
            # get next action
            new_action = epsilon_greedy(Q,new_state,nA,epsilon)
            
            # TD update
            # td_target
            td_target = reward+gamma*Q[new_state][new_action]
            # td_error
            td_error = td_target-Q[obs][action]
            # new Q
            Q[obs][action] += alpha*td_error
            # update state
            obs = new_state
            # update action
            action = new_action
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for episode_i in range(n_episodes):
        # initialize the environment 
        env.render()
        obs = env.reset()
        # loop for each step of episode
        done=False
        while not done:
            # get an action from policy
            action = epsilon_greedy(Q,obs,nA,epsilon=epsilon)
            # return a new state, reward and done
            new_state,reward,done,_=env.step(action)
            # TD update
            # td_target with best Q
            best_q = Q[new_state][np.argmax(Q[new_state])]
            td_target = reward+gamma*best_q
            # td_error
            td_error = td_target - Q[obs][action]
            # new Q
            Q[obs][action]+=alpha*td_error
            # update state
            obs = new_state
    ############################
    return Q

################################# RESULTS ##########################################
# ------Temporal Difference(50 points in total)------ ... ok
# epsilon_greedy (0 point) ... ok
# SARSA (25 points) ... ok
# Q_learning (25 points) ... ok

# ----------------------------------------------------------------------
# Ran 4 tests in 30.715s

# OK
