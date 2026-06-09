# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:56:51 2024

@author: pjan9
"""

import time
import math
import os
import random as rd
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

def vec_to_csv (vec):
    text = ''
    for i in vec:
        text += i + ','
    text = text[:-1]
        
def dist_sort (dists, rels):
    
    # Combine both lists into pairs
    combined = list(zip(dists, rels))
    
    # Sort the combined list by distances (the first element of each pair)
    combined.sort(key=lambda x: x[0])
    
    # Unpack the sorted pairs back into two lists
    sorted_dists, sorted_rels = zip(*combined)
    
    # Convert the result back to lists (zip returns tuples)
    sorted_dists = list(sorted_dists)
    sorted_rels = list(sorted_rels)
    
    return sorted_dists, sorted_rels

def rew_sort (dists, rels, rews):
    
    # Combine both lists into pairs
    combined = list(zip(rews, dists, rels))
    
    # Sort the combined list by distances (the first element of each pair)
    combined.sort(key=lambda x: x[0])
    
    # Unpack the sorted pairs back into two lists
    sorted_rews, sorted_dists, sorted_rels = zip(*combined)
    
    # Convert the result back to lists (zip returns tuples)
    sorted_rews = list(sorted_rews)
    sorted_dists = list(sorted_dists)
    sorted_rels = list(sorted_rels)
    
    return sorted_dists, sorted_rels, sorted_rews

def normang(ang, mode: bool = False):
    
    #ang %= (math.pi*2)
    while ang > math.pi *2:
        ang -= math.pi * 2
    while ang < 0:
        ang += math.pi * 2
    
    if mode == False and ang > math.pi:
        ang -= 2 * math.pi

    return ang
        
#from pp_utils import Predator, Prey, Landmark
class Predator():
    def __init__(self, motivation: int = 1, nr: float = 0.,
                 maxspeed: int = 1, maxturn: float = math.pi,
                 x_pos: int = 0.0, y_pos: int = 0.0, ori: float|None = 0.0,
                 angular_velocity: float = 0.0, linear_velocity: float = 0.0,
                 radius: int = 8.5,
                 observation: list = [] #check data type!
                 ):
        
        self.motivation = motivation
        self.maxspeed = maxspeed * self.motivation
        self.maxturn = maxturn
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.ori = ori
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.radius = radius
        self.observation = observation
        self.nr = nr
        
        #do i need to keep track of obs here?
    
    def move_simple(self, action):
        
        self.x_pos += action[0] * self.maxspeed
        self.y_pos += action[1] * self.maxspeed
        
    def move(self, action):
            
        self.angular_velocity = self.maxturn * action[0] #* self.motivation
        self.linear_velocity = self.maxspeed  * action[1] #* self.motivation
        
        #self.angular_velocity = self.observation[4]
        self.ori += self.angular_velocity
        self.ori = normang(self.ori, True) ###normalizing -pi, pi, should change to 0, 2pi
        
        #self.linear_velocity = self.maxspeed
        self.x_pos += math.cos(self.ori)  * self.linear_velocity
        self.y_pos += math.sin(self.ori)  * self.linear_velocity
        
        """idea: add collisions"""

class Prey(): #non-agent
    def __init__(self, maxspeed: int = 10, maxturn: float = math.pi,
                 x_pos: int|None = None, y_pos: int|None = None, ori: float|None = None,
                 angular_velocity: float = 0.0, linear_velocity: float = 0.0,
                 radius: int = 8.5,
                 max_x: bool|int = 0, max_y: bool|int = 0,
                 margin: bool|int = 50, horizon: int = 5,
                 observation: list = [], #check data type!
                 caught: bool = False,
                 lag: int | None = None
                 ):
        
        self.maxspeed = maxspeed
        self.maxturn = maxturn
        self.radius = radius
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.ori = ori
        
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

        self.observation = observation
        self.goal = None
        
        self.margin = margin
        self.horizon = horizon
        self.lag = lag
        if self.lag == None: self.lag = horizon
        self.counter = 0
        
        self.caught = caught
            
    def update_goal(self):
        
        #""" add ghost predators for arena edges
        #left boundary
        wall_x = min(-self.max_x/2, self.x_pos - self.margin)
        wall_y = self.y_pos
        wall_d = 'Left/West'
        
        vector_x = wall_x - self.x_pos
        vector_y = wall_y - self.y_pos
        
        relang = np.arctan2(vector_y, vector_x) 
        relang = normang(relang, True)
        
        dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
        
        self.observation.append([relang, wall_x, wall_y, wall_d, dist])
        
        #right boundary
        wall_x = max(self.max_x/2, self.x_pos + self.margin) 
        wall_y = self.y_pos
        wall_d = 'Right/East'
        
        vector_x = wall_x - self.x_pos
        vector_y = wall_y - self.y_pos
        
        relang = np.arctan2(vector_y, vector_x)
        relang = normang(relang, True)
        
        dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
        
        self.observation.append([relang, wall_x, wall_y, wall_d, dist])
     
        #down boundary
        wall_x = self.x_pos
        wall_y = max(self.max_y/2, self.y_pos + self.margin)
        wall_d = 'Down/South'
        
        vector_x = wall_x - self.x_pos
        vector_y = wall_y - self.y_pos
        
        relang = np.arctan2(vector_y, vector_x)
        relang = normang(relang, True)
        
        dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
        
        self.observation.append([relang, wall_x, wall_y, wall_d, dist])
        
        #up boundary
        wall_x = self.x_pos
        wall_y = min(-self.max_y/2, self.y_pos - self.margin) 
        wall_d = 'Up/North'
        
        vector_x = wall_x - self.x_pos
        vector_y = wall_y - self.y_pos
        
        relang = np.arctan2(vector_y, vector_x)
        relang = normang(relang, True)
        
        dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
        
        self.observation.append([relang, wall_x, wall_y, wall_d, dist])
        #"""
        
        #predicted trajectories
        turn = 0
        done = False  
        dists = []
        oris = []
        
        limit = normang(359*math.pi/180, True)
        
        while done == False:
            
            if turn > limit:
                turn = limit
            
            oris.append(turn)
            
            final_x = self.x_pos
            final_y = self.y_pos
            
            n_steps = self.horizon
            for i in range (n_steps):
                final_x += math.cos(turn) * self.linear_velocity
                final_y += math.sin(turn) * self.linear_velocity
            
            new_dist = np.inf
            
            for o in range( len (self.observation)):
                vector_x = self.observation[o][1] - final_x
                vector_y = self.observation[o][2] - final_y
                
                dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                
                if dist < new_dist:
                    new_dist = dist
                
            dists.append(new_dist)
            
            
            if turn >= limit:
                done = True
            else:
                turn += 360/20 * (math.pi/180)
                #turn = normang(turn, True)
        
        #choose new ori
        paired = list(zip( dists, oris) )
        rd.shuffle(paired)
        goal = max(paired, key=lambda x: x[0])[1]
        
        """implement mod scores based on how close from current goal"""
        #don't update goal if too close to current direction (use mids)
                
        return goal
    
    def move(self):
        
        #update speed
        min_dist = min(d[4] for d in self.observation) #find closest entity
        
        if min_dist >= self.max_x/2:
            self.linear_velocity = 0
            
        elif min_dist <= self.max_x/4:
            self.linear_velocity = self.maxspeed
            
        else:
            self.linear_velocity = self.maxspeed * (self.max_x / 2 - min_dist) / (self.max_x / 2 - self.max_x / 4)
        
        """
        #add noise to speed
        noise = rd.randrange(95, 101, 1) / 100
        self.linear_velocity *= noise
        """
        
        #update goal
        
        if self.counter >= self.lag or self.goal == None:
            self.goal = self.update_goal()
            self.counter = 0
        else:
            self.counter += 1
        
        # Calculate the shortest difference to the goal
        diff = (self.goal - self.ori + math.pi) % (2 * math.pi) - math.pi
    
        # Turn in the direction of the shortest angular difference
        if diff > self.maxturn:
            self.ori += self.maxturn
        elif diff < -self.maxturn:
            self.ori -= self.maxturn
        else:
            # If the goal is within maxturn, set ori to the goal
            self.ori = self.goal
        
        #self.ori = self.goal
        self.ori = normang(self.ori, True)
        
        #update position
        self.x_pos += math.cos(self.ori) * self.linear_velocity
        self.y_pos += math.sin(self.ori) * self.linear_velocity
        
        """idea: add collisions"""
    
class Landmark():
    def __init__(self,
                 x_pos: int|None = None, y_pos: int|None = None,
                 radius: int = 16):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius

class SeqPredPreyEnv(gym.Env):
    metadata = {
        "name":["Sequential_Predator_Prey_Environment"],
        "render_modes":["human"],
        "render_fps": 50,
        "is parallelizable" : True
    }
    
    def __init__(self,
                 render_mode: str|None = None,
                 record: str | None = None,
                 num_pred: int = 3,
                 num_prey: int = 1,
                 num_landmark: int = 0,
                 coop: bool = True,
                 var: bool = False,
                 name: str|None = None,
                 see_ally: bool = True,
                 options: dict = {}
                 ):
        
        #initialize trackers
        self.num_eps = 0 #keep track of episode count
        
        self.times = [] #keep track of total timestep since env initialization
        self.tot_rew = 0 #keep track of rew since env initialization
        self.terms = 0 #keep track of termination since env initialization
        
        #keep track of performance in last tol eps for current speed
        self.tol = options.get('tol', 20)
        self.stack = [0]*self.tol
        self.logger = 0
        
        self.SCORES = {f'pred{a}': [] for a in range(num_pred)} #track rew p/ agent p/ ep
        
        self.speeds = {f'prey{p}': [] for p in range(num_prey)} #track prey speed p/ prey p/ ep
        
        self.coop = coop #set the env to global swarm reward or single agent local reward
        
        self.var = var
          
        self.see_ally = see_ally
        
        #initialize spaces
        self.num_pred = num_pred
        self.num_prey = num_prey
        self.num_landmarks = num_landmark
        
        self.possible_agents = [f'pred{i}' for i in range(self.num_pred)]
           
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        
        obs_size = 0
        if self.see_ally == True:
            obs_size += self.num_pred * 2 +1 #perceive self & partners
        else:
            obs_size += 3 #perceive only self
        
        if self.var == True: obs_size += 1 #perceive own motivation
        
        obs_size += self.num_prey * 2 #perceive prey
        
        #obs_size += self.num_landmarks * 2 #obs landmarks (unused, landmarks tba)
        
        self.observation_space = spaces.Box(-1., 1., shape=(obs_size,), dtype='float32')
            
        #initialize dynamic options
        self.options = options
        
        #track if training
        self.name = name
        
        #initialize video options
        self.render_mode = render_mode
        self.frames = []
        self.record = record
        self.screen = None
        self.scale = options.get('scale', 1)
        """make render handle scale"""

    def place_entities(self):
        
        """idea: allow options to be passed to manually set position of each entity
        increase spawn area
        """
        
        for a in self.agents:
            self.agents[a].x_pos = rd.randrange(-int(self.max_x/4), int(self.max_x/4) +1, 1)
            self.agents[a].y_pos = rd.randrange(-int(self.max_y/4), int(self.max_y/4) +1, 1)
            self.agents[a].ori = rd.randrange(0, 361) * math.pi/180
        
        for a in self.preys:
            
            ok = 0
            while ok < self.num_pred:
                
                ok = 0
                self.preys[a].x_pos = rd.randrange(-int(self.max_x/4), int(self.max_x/4) +1, 1)
                self.preys[a].y_pos = rd.randrange(-int(self.max_y/4), int(self.max_y/4) +1, 1)
                
                for b in self.agents:
                    vector_x = self.preys[a].x_pos - self.agents[b].x_pos
                    vector_y = self.preys[a].y_pos - self.agents[b].y_pos
                    
                    dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                    
                    if dist > 3 * self.agents[b].maxspeed: #self.agents[b].maxspeed: #make sure prey isnt catchable in 1st step #50
                        ok += 1
                        
                    
            self.preys[a].ori = rd.randrange(0, 361) * math.pi/180
        
        """for a in self.landmarks:
            
            done = False
            while done == False:
                
                self.landmarks[a].x_pos = rd.randrange(-100, 101, 1)
                self.landmarks[a].y_pos = rd.randrange(-100, 101, 1)
                
                for b in self.agents:
                    vector_x = self.landmarks[a].x_pos - self.agents[b].x_pos
                    vector_y = self.landmarks[a].y_pos - self.agents[b].y_pos
                    
                    dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                    
                    if dist > self.agents[b].maxspeed: #make sure landmark is not reachable in 1 step
                        
                        for c in self.preys:
                            
                            vector_x = self.landmarks[a].x_pos - self.preys[c].x_pos
                            vector_y = self.landmarks[a].y_pos - self.preys[c].y_pos
                            
                            dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                            
                            if dist > self.preys[c].maxspeed: #also for preys
                                done = True"""
            
    def observe_pred(self, agent):
        #[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`
        
        """idea: occlusion, landmarks block perception. do other predators and preys also do it?"""
        """idea: perceive communication or motivation of other agents"""
        """idea: remove caught prey from observation to make it work with multiple preys"""
        
        observation = []
        
        #OBSERVE SELF
        #observation.append(self.agents[agent].linear_velocity)
        observation.append(self.agents[agent].x_pos / (self.max_x/2))
        observation.append(self.agents[agent].y_pos / (self.max_y/2))
        
        #observation.append(self.agents[agent].angular_velocity)
        observation.append(self.agents[agent].ori / (math.pi*2) )
        
        if self.var == True: observation.append(self.agents[agent].nr / max(1, (self.num_pred -1) ) )
        
        #OBSERVE PREYS
        preys_dists = []
        preys_rels = []
        for p in self.preys:
            
            vector_x = self.preys[p].x_pos - self.agents[agent].x_pos
            vector_y = self.preys[p].y_pos - self.agents[agent].y_pos
            
            relang = math.atan2(vector_y, vector_x)
            relang = relang - self.agents[agent].ori
            
            relang = normang(relang, False) / (math.pi)
            
            #relang = (relang + math.pi) % (2 * math.pi) - math.pi #normalize in -pi : pi
            #relang /= math.pi #normalize in -1 : 1
            
            dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
            
            dist = 1 - dist/self.max_dist # normalize in 0, 1
            if dist > 1.: dist = 1.
            if dist < 0.: dist = 0. #clip
                
            preys_dists.append(dist)
            preys_rels.append(relang)
            
            #observation.append(self.preys[p].x_pos)
            #observation.append(self.preys[p].y_pos)
        
        """#sort preys by dist to agent
        if len(self.preys) > 0:
            preys_dists, preys_rels = dist_sort(preys_dists, preys_rels)
        """

        observation += preys_dists
        observation += preys_rels
        
        #OBSERVE OTHER PREDATORS
        if self.see_ally:
            preds_dists = []
            preds_rels = []
            preds_topreys = []
            for a in self.agents:
                if a != agent:
                    #observation.append(self.agents[a].x_pos -self.agents[agent].x_pos)
                    #observation.append(self.agents[a].y_pos -self.agents[agent].y_pos)
                    
                    #av.append(self.agents[a].linear_velocity)
                    vector_x = self.agents[a].x_pos - self.agents[agent].x_pos
                    vector_y = self.agents[a].y_pos - self.agents[agent].y_pos
                    
                    relang = math.atan2(vector_y, vector_x)
                    relang = relang - self.agents[agent].ori
                    
                    relang = normang(relang, False) / (math.pi)
                    
                    #relang = (relang + math.pi) % (2 * math.pi) - math.pi #normalize in -pi : pi
                    #relang /= math.pi #normalize in -1 : 1
                    
                    dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                    
                    dist = 1 - dist/self.max_dist #normalize in 0, 1
                    if dist > 1.: dist = 1.
                    if dist < 0.: dist = 0. #clip
                    
                    dist_to_preys = 0
                    for b in self.preys:
                        dist_to_preys += math.sqrt( (self.agents[a].x_pos - self.preys[b].x_pos) ** 2 + (self.agents[a].y_pos - self.preys[b].y_pos) ** 2)
                    dist_to_preys /= len(self.preys)
                    dist_to_preys *= -1
                    
                    #observation.append(self.agents[a].x_pos)
                    #observation.append(self.agents[a].y_pos)
                    #observation.append(dist)
                    #observation.append(relang)
                    
                    preds_dists.append(dist)
                    preds_rels.append(relang)
                    preds_topreys.append(dist_to_preys)
        
            """
            #sort preds by dist to prey
            if len(self.agents) > 2:
                preds_dists, preds_rels, preds_topreys = rew_sort(preds_dists, preds_rels, preds_topreys)
            """
        
            observation += preds_dists
            observation += preds_rels
        
        """OBSERVE LANDMARKS
        landmarks_dists = []
        landmarks_rels = []    
        for l in self.landmarks:
            #observation.append(self.landmarks[l].x_pos -self.agents[agent].x_pos)
            #observation.append(self.landmarks[l].y_pos -self.agents[agent].y_pos)
            
            vector_x = self.landmarks[l].x_pos - self.agents[agent].x_pos
            vector_y = self.landmarks[l].y_pos - self.agents[agent].y_pos
            
            relang = math.atan2(vector_y, vector_x)
            relang = relang - self.agents[agent].ori
            
            relang = (relang + math.pi) % (2 * math.pi) - math.pi #normalize in -pi : pi
            relang /= math.pi #normalize in -1 : 1
            
            dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
            
            dist = 1 - dist/self.max_dist # normalize in 0, 1
            if dist > 1.: dist = 1.
            if dist < 0.: dist = 0. #clip
            
            landmarks_dists.append(dist)
            landmarks_rels.append(relang)
            
        #sort landmarks by dist to agent
        #if len(self.landmarks) > 0: 
            #landmarks_dists, landmarks_rels = dist_sort(landmarks_dists, landmarks_rels)
            
        observation += landmarks_dists
        observation += landmarks_rels
        """
        
        observation = np.array(observation, dtype=np.float32)
        self.agents[agent].observation = observation
        
        return observation
    
    def observe_prey(self, prey):
        
        observation = []
        for entity in self.agents:
            
            vector_x = self.agents[entity].x_pos - self.preys[prey].x_pos
            vector_y = self.agents[entity].y_pos - self.preys[prey].y_pos
            
            relang = np.arctan2(vector_y, vector_x) #- self.ori
            relang = normang(relang, True)
            
            dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
            
            observation.append([relang,
                                self.agents[entity].x_pos, self.agents[entity].y_pos,
                                True, dist])
        
        for entity in self.preys:
            if prey != entity:
                vector_x = self.preys[entity].x_pos - self.preys[prey].x_pos
                vector_y = self.preys[entity].y_pos - self.preys[prey].y_pos
                
                relang = np.arctan2(vector_y, vector_x) #- self.ori
                relang = normang(relang, True)
                
                dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
                
                observation.append([relang,
                                    self.preys[entity].x_pos, self.preys[entity].y_pos,
                                    False, dist])
        
        """for entity in self.landmarks:
            vector_x = self.landmarks[entity].x_pos - self.preys[prey].x_pos
            vector_y = self.landmarks[entity].y_pos - self.preys[prey].y_pos
            
            relang = np.arctan2(vector_y, vector_x) #- self.ori
            relang = normang(relang, True)
            
            dist = math.sqrt( vector_x ** 2 + vector_y ** 2)
            
            observation.append([relang,
                                self.landmarks[entity].x_pos, self.landmarks[entity].y_pos,
                                True, dist])"""
            
        """idea: occlusions"""
        
        return observation
        
            
    def reset(self, seed: int|None = None, options: dict = {}):
        
        """increase time limit"""
        # Seed setting for reproducibility
        
        if options is None:
            options = {}
            
        seed = options.get('seed', seed)
        
        if seed is not None:
            np.random.seed(seed)
            rd.seed(seed)
            #self.seed = seed
            
        self.options = self.options | options
        
        #initialize clocks
        self.timestep = 0 #initialize clock
        self.agent_iter = 0 #initialize actor tracker
        
        self.max_steps = self.options.get('max_steps', 500)
        
        #initialize arena
        self.max_y = self.options.get('max_y', 700)
        self.max_x = self.options.get('max_x', 700)
        self.max_dist = math.sqrt( self.max_x**2 + self.max_y**2)
        
        #initialize predators
        min_motivation = float(self.options.get('min_motivation', .9) )
        max_motivation = float(self.options.get('max_motivation', 1. + (1 - min_motivation) ) )
        
        flip = self.options.get('soft_flip', False)

        # Four explicit evaluation modes:
        # var=False, flip=False  -> EQUAL (native same)
        # var=True,  flip=False  -> VARIED (native varied)
        # var=False, flip=True   -> VARIED (flipped same -> varied)
        # var=True,  flip=True   -> EQUAL  (flipped varied -> equal)
        if bool(self.var) == bool(flip):
            # (var is True and flip is True) or (var is False and flip is False)
            min_motivation, max_motivation = 1.0, 1.0
            
        else:
            if min_motivation == max_motivation:
                print('min and max motivation were the same, resetting to default min=.9 max =1.1')
                min_motivation = .9
                max_motivation = 1.1
            elif min_motivation > max_motivation:
                print('min mot was higher than max mot, these were auto flipped')
                min_motivation, max_motivation = max_motivation, min_motivation
            
        mot_step = (max_motivation - min_motivation) / max( (self.num_pred -1), 1 )

        self.agents = {}
        for a in range (self.num_pred):
            
            motivation = np.float32(min_motivation + mot_step * a) #fixed motivation ensuring equally sparsed between (and including) min and max motivation
            """idea: get a random motivation between min and max"""
            
            self.agents[f'pred{a}'] = Predator(motivation = motivation,
                                               nr = a,
                                              maxspeed = self.options.get('pred_max_speed', 10),
                                              maxturn = self.options.get('pred_max_turn', math.pi/6),
                                              radius = self.options.get('pred_rad', 8.5))

        #initialize preys
        self.prey_max_speed = self.options.get('new speed', self.options.get('prey_max_speed', 0))
        
        self.preys = {}
        
        for a in range (self.num_prey):
            ori = rd.randrange(0, 361, 1)
            ori *= math.pi/180
            ori = normang(ori, True)
            prey_speed = self.prey_max_speed #* rd.randrange(90, 100)/100
            self.preys[f'prey{a}'] = Prey(maxspeed = prey_speed, #*rd.random(0.85, 1) assign varying speeds however make sure max speed is used often
                                          maxturn = self.options.get('prey_max_turn', self.options.get('pred_max_turn', math.pi/6) ),
                                         radius = self.options.get('prey_rad', self.options.get('pred_rad', 8.5)),
                                         lag = self.options.get('prey_lag', 0),
                                         max_x = self.max_x,
                                         max_y = self.max_y)
            
        """#initialize landmarks 
        landmark_rad = self.options.get('landmark_rad', self.options.get('pred_rad', 8.5)*2)
        
        self.landmarks = {}
        for l in range (self.num_landmarks):
            self.landmarks[f'landmark{l}'] = Landmark(radius = landmark_rad)
        #"""
        
        #place agents and objects
        """idea: configs from options to be passed here"""
        self.place_entities()
        
        #initialize agent state
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.ep_rews = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.observations = {a: (self.observe_pred(a)) for a in self.agents}
        
        self.truncated = False
        self.terminated = False
        self.reward = 0.
        
        self.observation = self.observations[f'pred{self.agent_iter}']
        #terminated = self.terminations[f'pred{self.agent_iter}']
        #truncated = self.truncations[f'pred{self.agent_iter}']
            
        #infos is also a dict of tuple p/ agent
        self.infos = {a: {} for a in self.agents}
        self.info = {}
        """make info return some useful data"""
            
        if self.render_mode == 'human':
            self.render()
        
        #time.sleep(1)
        return self.observation, self.info
            
    def compute_rewards(self, agent):
        
        reward = 0
        
        r = []
        for prey in self.preys:
            r.append ( math.sqrt( (self.agents[agent].x_pos - self.preys[prey].x_pos)**2 +
                               (self.agents[agent].y_pos - self.preys[prey].y_pos)**2 ) )
        
        for i in range ( len (r)):
            if r[i] <= self.preys[f'prey{i}'].radius + self.agents[agent].radius and self.preys[f'prey{i}'].caught == False:
                self.preys[f'prey{i}'].caught = True
                reward += 1
                
        punish = 0
        
        #penalize agents for moving out of arena
        if abs(self.agents[agent].x_pos) > self.max_x:
            #reward -= abs(self.agents[agent].x_pos) - self.max_x
            punish = -0.01
        
        elif abs(self.agents[agent].y_pos) > self.max_y:
            #reward -= abs(self.agents[agent].y_pos) - self.max_y
            punish = -0.01
        
        return reward + punish
    
    def logg(self):
        
        self.num_eps += 1
            
        if self.logger >= self.tol:
            self.logger = 0
            
        if self.terminated:
            self.stack[self.logger] = 1
            self.terms += 1
        else:
            self.stack[self.logger] = 0
            
        for agent in self.ep_rews: #adds to SCORES the rew of finishing ep
            self.SCORES[agent].append(self.ep_rews[agent])
        
        for p in self.preys: #adds to speeds to current prey speed
            self.speeds[p].append( self.preys[p].maxspeed )
        
        self.times.append( self.timestep  )
        
        self.catch_rate = sum(self.stack) / self.tol #need for the callback
            
        self.logger += 1
            
    def check_truncation(self):
        
        if self.timestep >= self.max_steps:
            return True
        
        return False
    
    def check_terminated(self):
        
        caught = 0
        for p in self.preys:
            if self.preys[p].caught == True:
                caught += 1
                
        if caught >= self.num_prey:
            return True
        
        return False
        
    def step(self, action):
        
        self.truncated = False
        self.terminated = False
        
        #Find acting agent
        actor = f'pred{self.agent_iter}'
        next_actor = f'pred{self.agent_iter + 1}'
        
        if self.agent_iter == len(self.agents)-1:
            next_actor = 'pred0'
        
        #Move acting agent
        self.agents[actor].move(action)
        
        self.rewards[actor] = self.compute_rewards(actor)
        #self.rewards[actor] = self.compute_reward(actor)
        
        self.agent_iter += 1
        
        #Update env once all agents moved
        if self.agent_iter == len(self.agents):
            
            #update trackers
            self.agent_iter = 0
            self.timestep += 1
            
            for prey in self.preys:
                
                if self.preys[prey].caught == False:
                    
                    self.preys[prey].observation = self.observe_prey(prey)
                    
                    self.preys[prey].move()
                
            # Update observations for all agents
            self.observations = {a: self.observe_pred(a) for a in self.agents}
            
        self.terminations[actor] = self.check_terminated()
        self.truncations[actor] = self.check_truncation()
        
        #from agent states retrieve env state
        self.terminated = self.terminations[actor]
        self.truncated = self.truncations[actor]
            
        self.info = {} #make info useful
        
        self.observation = self.observations[next_actor]
        
        self.reward = self.rewards[actor]
        
        if self.render_mode == "human":
            self.render()

        self.ep_rews[actor] += self.rewards[actor] #keep track of total reward p/ agent in episode
        
        if self.terminated or self.truncated: #keep track of rewards p/ episode
            self.logg()
            
        self.tot_rew += self.reward
        
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def render(self):
        import pygame
        
        d = -np.inf
        for l in [self.agents, self.preys]:
            for p in l:
                x = max(l[p].x_pos, l[p].y_pos)
                d = max(d, x)
        
        self.scale = 1
        
        if d == 0: scale = self.scale
        
        else:
            scale = min( self.scale, abs(self.max_x*.5 / d), abs(self.max_y*.5 / d) )
        
        def conv(pt, sw = self.max_x, sh = self.max_y, scale = scale):
            #adjusts coords to pygame referential
            new = [0,0]
            new[0] = pt[0] * scale + sw / 2
            new[1] = pt[1] * scale + sh / 2
                
            return new
        
        #set up screen
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode([self.max_x, self.max_y], pygame.RESIZABLE)

        # Fill the background with white
        self.screen.fill((255, 255, 255))
        
        #set font for obs display
        font = pygame.font.Font(None, 12)
        
        """
        #render prey perception
        for prey in self.preys:
            for o in range( len ( self.preys[prey].observation )):
                x = self.preys[prey].observation[o][1]
                y = self.preys[prey].observation[o][2]
                pygame.draw.circle(self.screen, (0.25*255, 0.75*255, 0.75*255), conv([x,y]), self.preys[prey].radius*scale*2)
        #"""
        
        #render preys
        for p in self.preys:
            x, y = self.preys[p].x_pos, self.preys[p].y_pos
            pygame.draw.circle(self.screen, (0.25*255, 0.75*255, 0.25*255), conv([x,y]), self.preys[p].radius*scale)
        
        #render predators
        for a in self.agents:
            x, y = self.agents[a].x_pos, self.agents[a].y_pos
            color = 0.75*255, 0.25*255, 0.25*255
            pygame.draw.circle(self.screen, color, conv([x,y]), self.agents[a].radius*scale)
        
        """
        #render landmarks
        for l in self.landmarks:
            x, y = self.landmarks[l].x_pos, self.landmarks[l].y_pos
            pygame.draw.circle(self.screen, (0.25*255, 0.25*255, 0.25*255), conv([x,y]), self.landmarks[l].radius*scale)
        #"""
        
        """
        #draw initial posotion area
        ix, iy = conv([(0 -self.max_x/4), (0 - self.max_y/4)])
        pygame.draw.rect(self.screen, (255,0,0), pygame.Rect(ix, iy, self.max_x/2*scale, self.max_y/2*scale),  2)
        #"""
        
        """
        # Render the text (split by lines)
        lines = []
        for a in self.agents:
            lines.append(str(self.agents[a].observation))
        for p in self.preys:
            lines.append(str(self.preys[p].linear_velocity))
        
        lines.append( str(scale) )
            
        # Set starting position
        x, y = 100, 100
        line_height = font.get_linesize()  # You can also use a fixed value
        
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (x, y + i * line_height))
        #"""
        
        pygame.display.flip()
        pygame.event.pump()
        
        if self.record != None:
            # Capture the screen and store the frame
            frame = pygame.surfarray.array3d(self.screen)
            # Convert from (width, height, 3) to (height, width, 3) because pygame's coordinate system is different
            frame = frame.transpose((1, 0, 2))
            self.frames.append(frame)
    
    def save_video(self, video_filename="seqpredprey.mp4", fps=30):
        
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        
        # Create a /videos directory if it doesn't exist
        video_dir = "videos"
        os.makedirs(video_dir, exist_ok=True)
    
        # Full path to save the video
        video_path = os.path.join(video_dir, video_filename)
    
        if hasattr(self, 'frames') and self.frames:
            # Create a video from the stored frames
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(video_path, codec='libx264')
        else:
            print("No frames to save as video.")
            
    def close(self):
            
        if self.record != None:
            self.save_video(f'{self.record}.mp4')
            
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            
    #check gym doc to see the best spaces
    def get_observation_space(self, agent):
        return self.observation_spaces[agent]

    def get_action_space(self, agent):
        return self.action_spaces[agent]

def env(**kwargs):
    env = SeqPredPreyEnv(**kwargs)
    env.reset()
    return env

if __name__=="__main__":
    pass