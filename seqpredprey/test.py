# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:51:38 2025

@author: pjan9
"""

import gymnasium as gym
import seqpredprey

from stable_baselines3 import PPO

import time

scn = "SeqPredPreyEnv-v0"

render_mode = None
record = None
wait = 0.

num_games = 1

speeds = [10,11,12,13,14,15,16,17,18,19,20]
num_preds = [2,3]

NS = 3
conds = ['cs','ss','sv','cv']

for cond in conds:
    if cond[0] == 'c':
        coop = True
    else: coop = False
    
    if cond[1] == 'v':
        var = True
        min_mot = .9
    else:
        var = False
        min_mot = 1
        
    if len(cond) >= 3 and cond[2] == 'b':
        sa = True
    else: sa = True
    
    for ns in range(1, NS+1):
        
        name = cond+ns 
        
        for num_pred in num_preds:
            for speed in speeds:
                config = {'prey_max_speed': speed,
                         'min_motivation': min_mot,
                         'tol': 20,
                         'name': name,
                         }
        
                model = PPO.load(f"{name}")
                env = gym.make("SeqPredPreyEnv-v0", render_mode=render_mode, record=record, name = name,
                               options = config, var=var, coop=coop, sa= sa, num_pred = num_pred)
                Rews = 0
                REWS = 0
                goals = 0
                
                g = 0
                while g < num_games:
                
                    obs, info = env.reset()
                    done = False
                
                    while not done:
                
                        action = model.predict(obs, deterministic=True)[0]
                        obs, rew, term, trunc, info = env.step(action)
                        time.sleep(wait)
                        Rews += rew
                
                        if term or trunc:
                            REWS += Rews
                            if term:
                                goals += 1
                
                            print(f'game: {g}, reward: {Rews}, terminated: {term}')
                
                            g += 1
                            Rews = 0
                            done = True
                
                print('Average Reward: ', REWS/num_games)
                print('Catch Rate: ', goals/num_games)
                env.close()