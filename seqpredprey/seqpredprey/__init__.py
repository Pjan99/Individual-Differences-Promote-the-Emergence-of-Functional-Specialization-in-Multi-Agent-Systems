# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 02:13:56 2024

@author: pjan9
"""

from gymnasium.envs.registration import register
register(
    id='SeqPredPreyEnv-v0',
    entry_point='seqpredprey.envs:SeqPredPreyEnv',)