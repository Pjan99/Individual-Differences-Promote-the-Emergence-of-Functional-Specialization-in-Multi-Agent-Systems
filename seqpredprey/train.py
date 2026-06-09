# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:50:59 2024

@author: pjan9
"""

import argparse
import gymnasium as gym
import time
import torch as th
import numpy as np
import random as rd
import copy
import os
import pandas as pd

import seqpredprey

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

def save_buffer_csv(buffer, filepath):
    # Prepare dictionary of arrays for each data type
    data_dict = {}

    # Helper to flatten a list of arrays/tensors into 2D numpy array
    def flatten_list_of_arrays(lst):
        arrs = []
        for x in lst:
            # Convert tensor to numpy if needed
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            # Flatten multidimensional arrays to 1D
            arrs.append(np.array(x).flatten())
        return np.vstack(arrs)

    # Flatten observations, actions, rewards, episode_starts, values, log_probs
    data_dict["observations"] = flatten_list_of_arrays(buffer.observations)
    data_dict["actions"] = flatten_list_of_arrays(buffer.actions)
    data_dict["rewards"] = flatten_list_of_arrays(buffer.rewards)
    data_dict["episode_starts"] = flatten_list_of_arrays(buffer.episode_starts)
    data_dict["values"] = flatten_list_of_arrays(buffer.values)
    data_dict["log_probs"] = flatten_list_of_arrays(buffer.log_probs)

    # Returns and advantages are usually 1D tensors of length = buffer size
    # Convert to 2D arrays with shape (N, 1)
    data_dict["returns"] = np.array(buffer.returns).reshape(-1, 1)
    data_dict["advantages"] = np.array(buffer.advantages).reshape(-1, 1)

    # Now build a single dataframe by concatenating all columns with distinct names
    df = pd.DataFrame()

    for key, arr in data_dict.items():
        n_cols = arr.shape[1]
        col_names = [f"{key}_{i}" for i in range(n_cols)]
        df_part = pd.DataFrame(arr, columns=col_names)
        df = pd.concat([df, df_part], axis=1)

    # Save to CSV
    df.to_csv(filepath, index=False)

def save_rollouts_csv(agent_buffers, rollout_buffer, mixed_buffer, epoch, save_dir, save_every=100, max_saved=20):
    if epoch % save_every != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each agent's rollout as CSV
    for agent_id, buffer in agent_buffers.items():
        filepath = os.path.join(epoch_dir, f"{agent_id}_rollout.csv")
        save_buffer_csv(buffer, filepath)

    # Save full rollout as CSV
    full_filepath = os.path.join(epoch_dir, "full_rollout.csv")
    save_buffer_csv(rollout_buffer, full_filepath)
    
    # Save mixed rollout as CSV
    if mixed_buffer != None:
        full_filepath = os.path.join(epoch_dir, "mixed_rollout.csv")
        save_buffer_csv(mixed_buffer, full_filepath)
    
    # Remove older saved rollouts, keep only last max_saved
    all_dirs = sorted([d for d in os.listdir(save_dir) if d.startswith("epoch_")],
                      key=lambda x: int(x.split("_")[1]))
    if len(all_dirs) > max_saved:
        for old_dir in all_dirs[:-max_saved]:
            full_path = os.path.join(save_dir, old_dir)
            for file in os.listdir(full_path):
                os.remove(os.path.join(full_path, file))
            os.rmdir(full_path)
            
class CustomPPO(PPO):

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        
        Custom method to collect rollouts for multi-agent environments, ensuring experiences are ordered
        step1agent1, step2agent1, ..., stepNagent1, step1agent2, step2agent2, ..., stepNagent2...
        
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment Currently, only implemented for n_envs = 1 (or env.num_envs = 1)
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        
        assert env.num_envs == 1, "implemented only for n_envs = 1"
        
        assert self._last_obs is not None, "No previous observation was provided"
        
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        envu = env.envs[0].unwrapped
        
        last_obs = {}
        last_dones = {}
        buffers = {}
        
        to_update = []
        
        for a in envu.possible_agents:
            last_obs[a] = None
            last_dones[a] = None
            buffers[a] = {}
            buffers[a]['observations'] = []
            buffers[a]['actions'] = []
            buffers[a]['rewards'] = []
            buffers[a]['episode_starts'] = []
            buffers[a]['values'] = []
            buffers[a]['log_probs'] = []
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            actor = envu.agent_iter
            current_agent = f'pred{actor}'
            
            self._last_obs = np.array([ envu.observations[current_agent] ])
            last_obs[current_agent] = self._last_obs
            
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, gym.spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            if current_agent not in to_update:
                to_update.append(current_agent)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            if 'terminal_observation' in infos[0]: #make sure terminal obs passed to infos matches acting agent
                infos[0]['terminal_observation'] = last_obs[current_agent]
                
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                
            buffers[current_agent]['observations'].append( last_obs[current_agent] )
            buffers[current_agent]['actions'].append(actions)
            buffers[current_agent]['rewards'].append(rewards)
            buffers[current_agent]['episode_starts'].append( last_dones[current_agent] )
            buffers[current_agent]['values'].append(values.clone())
            buffers[current_agent]['log_probs'].append(log_probs.clone())
            
            # Handle timeout by bootstrapping with value function
            if (dones[0] == True
                and infos[0].get("terminal_observation", None) is not None
                and infos[0].get("TimeLimit.truncated", False)
            ):
                for a in to_update:
                    
                    terminal_obs = self.policy.obs_to_tensor(last_obs[a])[0]
                    
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    
                    buffers[a]['rewards'][-1][0] += self.gamma * terminal_value
                    buffers[a]['episode_starts'][-1] = dones
                    last_dones[a] = dones
            
            #share rewards across agents
            if envu.coop == True and dones[0] == True and infos[0].get('TimeLimit.truncated', False) == False:

                for a in to_update: #share only with agents that acted in this episode
                    if a != current_agent:
                        buffers[a]['rewards'][-1][0] += 1
                        buffers[a]['episode_starts'][-1] = dones
                        last_dones[a] = dones
                        
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            last_dones[current_agent] = self._last_episode_starts
            if dones[0] == True: to_update = []
                
        #turn dict buffers into rollout objects
        agent_buffers = {}
        for a in buffers:
            agent_buffer_size = len(buffers[a]['observations']) #find how many steps taken by agent
            
            #initialize rollout object
            agent_buffers[a] = RolloutBuffer(
                agent_buffer_size, env.observation_space, env.action_space, gamma=self.gamma,
                gae_lambda=self.gae_lambda, device=self.device, n_envs= 1 #self.n_envs
            )
            
            agent_buffers[a].reset()
            
            #add experiences to agent rollout
            for i in range(agent_buffer_size):
                
                agent_buffers[a].add(
                    buffers[a]['observations'][i],
                    buffers[a]['actions'][i],
                    buffers[a]['rewards'][i],
                    buffers[a]['episode_starts'][i],
                    buffers[a]['values'][i],
                    buffers[a]['log_probs'][i]
                )
                
            with th.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(obs_as_tensor( last_obs[a], self.device))  # type: ignore[arg-type]
                
                
            agent_buffers[a].compute_returns_and_advantage(last_values=values, dones=last_dones[a] )
    
        #concatenate agent rollouts in a single rollout
        stc = 0
        for a in agent_buffers:
            buffer = agent_buffers[a]
            
            for p in range( buffer.size() ):
                rollout_buffer.add(buffer.observations[p],
                                   buffer.actions[p],
                                   buffer.rewards[p],
                                   buffer.episode_starts[p],
                                   # Convert to PyTorch tensor
                                   th.tensor(buffer.values[p]),
                                   # Convert to PyTorch tensor
                                   th.tensor(buffer.log_probs[p]),
                                   )
    
                rollout_buffer.returns[stc+p] = buffer.returns[p]
                rollout_buffer.advantages[stc+p] = buffer.advantages[p]
            
            stc += buffer.size()

        callback.update_locals(locals())
        
        callback.on_rollout_end()
        
        #save rollouts
        current_epoch = self.num_timesteps // n_rollout_steps
        name = envu.options['name']
        save_dir = f'{name}_rollouts'
        
        return True

class CallBack(BaseCallback):

    def __init__(self, name, success_threshold: float = 0.9, speed_increment: float = 0.005, save_every = 5_000_000):
        
        super().__init__()
        
        self.name = name
        
        self.success_threshold = success_threshold  
        self.speed_increment = speed_increment   
        self.new_speed = 0
        self.checkpoint = 0

        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        
        self.epoch_rew = 0
        self.EPOCH_REWS = []
        
        self.epoch_eps = 0
        self.EPOCH_EPS = []
        
        self.epoch_speeds = []
        
        self.epoch_time = 0
        self.EPOCH_TIMES = []

        self.ACTOR_LOSSES = []
        self.CRITIC_LOSSES = []
        self.ENTROPIES = []
        self.CATCHES = []
        
        self.save_every = save_every
        self.last_low = 0
        self.last_update = 0
        
        try:
            with open(f'{self.name}_best_speed.txt', 'r') as data:
                best_speed = data.read()
        
            self.best_speed = np.float32(best_speed)
        except:
            self.best_speed = 0.
        
        print(self.best_speed)
        
    def _on_step(self) -> bool:
        #print('step: ', self.epoch_rews, env.tot_rew,
              #'\n', self.epoch_eps, env.num_eps)
        return True
    
    def _on_rollout_end(self) -> None:
        
        # Access the environment (assumes non-VecEnv)
        env = self.training_env.envs[0].unwrapped
        """make compatible with vec envs"""

        # Extracts actor loss, critic loss, and entropy from the logger and updates environment settings.

        try:
            # Get values from the logger (last recorded values)
            actor_loss = self.logger.name_to_value.get(
                "train/policy_loss", None)
            critic_loss = self.logger.name_to_value.get(
                "train/value_loss", None)
            entropy = self.logger.name_to_value.get("train/entropy_loss", None)

            if actor_loss is not None:
                self.actor_losses.append(actor_loss)
            if critic_loss is not None:
                self.critic_losses.append(critic_loss)
            if entropy is not None:
                self.entropies.append(entropy)

            # Compute averages
            avg_actor_loss = np.mean(
                self.actor_losses) if self.actor_losses else 0
            avg_critic_loss = np.mean(
                self.critic_losses) if self.critic_losses else 0
            avg_entropy = np.mean(self.entropies) if self.entropies else 0
            
            self.epoch_eps = copy.deepcopy(env.num_eps) -sum(self.EPOCH_EPS)
            self.epoch_rew = copy.deepcopy(env.tot_rew) -sum(self.EPOCH_REWS)
            
            avg_reward = self.epoch_rew / max(self.epoch_eps, 1)
             
            # store values across epochs
            self.ACTOR_LOSSES.append(avg_actor_loss)
            self.CRITIC_LOSSES.append(avg_critic_loss)
            self.ENTROPIES.append(avg_entropy)
            
            self.EPOCH_REWS.append(avg_reward)
            self.EPOCH_EPS.append(self.epoch_eps)
            self.CATCHES.append( env.catch_rate )
            
            self.epoch_time = sum(copy.deepcopy(env.times)) -sum(self.EPOCH_TIMES)
            self.EPOCH_TIMES.append(self.epoch_time)
            
            # Reset collected values for the next epoch
            self.actor_losses.clear()
            self.critic_losses.clear()
            self.entropies.clear()

        except Exception as e:
            print(f"Error extracting training metrics: {e}")

        # reset update flag #make compatible with vec env
        update = False
        #self.success_threshold = rd.randrange(0, 101, 1) / 100
        
        if (env.catch_rate >= self.success_threshold 
            and self.last_update >= env.tol
            ):
            
            self.checkpoint = env.prey_max_speed
            
            with open(f"{self.name}_preferred_speed.txt", "w") as data:
                data.write(str(self.checkpoint))
    
            # Increment prey speed
            if env.prey_max_speed +1 < 9.5:
                self.new_speed = env.prey_max_speed + 1
            else:
                self.new_speed = env.prey_max_speed + self.speed_increment
                
            self.new_speed = np.float32(self.new_speed)
            
            update = True
            env.options['new speed'] = self.new_speed

            print(f"Increased prey speed to {self.new_speed}")

            self.epoch_speeds.append( self.new_speed )
            
            self.last_update = 0
            
        elif (env.catch_rate < self.success_threshold *2/3
              and self.last_update >= env.tol
              ):
            

            if (env.prey_max_speed == self.checkpoint
                and env.catch_rate <= self.success_threshold /2
                and self.checkpoint >= 0
                ):
                
                self.checkpoint -= self.speed_increment
                
                if self.checkpoint < 0: self.checkpoint = 0
                
                self.checkpoint = np.float32(self.checkpoint)
                
                print(f'Decreased checkpoint to {self.checkpoint}')
                
            self.new_speed = env.prey_max_speed - self.speed_increment/5
            
            if self.new_speed < self.checkpoint:
                self.new_speed = self.checkpoint
                print(f"Prey speed stayed at {env.prey_max_speed}")
            
            else:
                print(f"Decreased prey speed to {self.new_speed}")
            
            self.new_speed = np.float32(self.new_speed)
            
            #env.options['speed update'] = True
            env.options['new speed'] = self.new_speed
            
            self.epoch_speeds.append(env.prey_max_speed )
            
            self.last_update = 0

        else:
            
            self.epoch_speeds.append(env.prey_max_speed )
            
            self.last_update += self.epoch_eps
            
            #print(f"Prey speed stayed at {self.new_speed}")
        
        print(f"Prey speed: {env.prey_max_speed}, catch rate: {env.catch_rate}")
        
        #Save logic
        """saves when 
        catch rate lower then half the avg of last 100 epochs
        periodically ever n (save_every) steps
        if speed is updated
        """
        
        self.last_low += self.epoch_eps
        if ( env.catch_rate < np.mean(self.CATCHES[-100:]) / 2
            and  env.catch_rate <= min(self.CATCHES[-50:])
            ):
            
            if len( self.CATCHES ) < 100:
                pass
            
            elif self.last_low < 100:
                pass
            
            else:
                stamp = f'_{int(self.num_timesteps/M)}M'
                save_name = self.name + f'_low' + stamp
                
                # Get a list of all files in the model directory
                model_dir = os.path.dirname(os.path.abspath(__file__))
                existing_files = os.listdir(model_dir)
                
                # Remove previous model files with the same name but different stamps
                for file in existing_files:
                    if file.startswith(self.name + '_low') and not file.endswith(stamp + '.zip'):
                        # Construct the full file path
                        file_path = os.path.join(model_dir, file)
                        print(f"Removing old model file: {file_path}")
                        os.remove(file_path)
                        
                # Save the new model
                self.model.save(save_name)
                print(f"Saved new model: {save_name}")
                
                self.last_low = 0
        
        #"""
        if self.num_timesteps % self.save_every == 0:
            self.model.save(self.name)
            self.save_data()
            print(f"Saved new model: {self.name}")
        #"""
            
        if update == True:
            #stamp = f'_{int(self.num_timesteps/M)}M'
            if self.new_speed > self.best_speed:
                save_name = self.name + '_best'
                self.model.save(save_name)
                self.best_speed = copy.copy(self.new_speed)
                
                t = str(self.best_speed)
                
                with open(f'{self.name}_best_speed.txt', 'w') as data:
                    data.write(t)
                    data.close()
                    
                print("Saved record speed model")
                
            save_name = self.name + '_update'
            self.save_data()
            self.model.save(save_name)
            print(f"Saved new model: {save_name}")
            
    def save_data(self):
        
        env = self.training_env.envs[0].unwrapped
        """make compatible with VecEnv"""
        
        #"""
        t = '\n'
        for a in env.SCORES:
            t += str(a) + '\n'
            for i in env.SCORES[a]:
                t += str(i) + ','
            t = t[:-1] + '\n'
        t = t[:-1]
        
        with open(f'{env.name}_scores.csv', 'a') as data:
            data.write(t)
            data.close()
        
        #clean data
        for a in env.SCORES:
            env.SCORES[a] = []
            
        t = '\n'
        for p in env.speeds:
            t += str(p) + '\n'
            for i in env.speeds[p]:
                t += str(i) + ','
            t = t[:-1] + '\n'
        t = t[:-1]
        
        with open(f'{env.name}_speeds.csv', 'a') as data:
            data.write(t)
            data.close()
        
        for p in env.speeds:
            env.speeds[p] = []
        #"""
        
        t = ''
        for ts in self.EPOCH_TIMES:
            t += str(ts) + ','
        
        with open(f'{env.name}_epoch_times.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.EPOCH_TIMES = []
        env.times = []
        
        t = ''
        for a in self.ACTOR_LOSSES:
            t += str(a) + ','
        
        with open(f'{env.name}_actor_losses.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.ACTOR_LOSSES = []
        
        t = ''
        for a in self.CRITIC_LOSSES:
            t += str(a) + ','
        
        with open(f'{env.name}_critic_losses.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.CRITIC_LOSSES = []
        
        t = ''
        for a in self.ENTROPIES:
            t += str(a) + ','
        
        with open(f'{env.name}_entropies.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.ENTROPIES = []
        
        t = ''
        for a in self.EPOCH_REWS:
            t += str(a) + ','
        
        with open(f'{env.name}_epoch_rewards.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.EPOCH_REWS = []
        env.tot_rew = 0
        
        t = ''
        for a in self.EPOCH_EPS:
            t += str(a) + ','
        
        with open(f'{env.name}_epoch_episodes.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.EPOCH_EPS = []
        env.num_eps = 0
        
        t = ''
        for a in self.epoch_speeds:
            t += str(a) + ','
        
        with open(f'{env.name}_epoch_speeds.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.epoch_speeds = []
        
        t = ''
        for a in self.CATCHES:
            t += str(a) + ','
        
        with open(f'{env.name}_epoch_catch_rate.csv', 'a') as data:
            data.write(t)
            data.close()
        
        self.CATCHES = []
        
        with open(f'{env.name}_final_speed','w') as data:
            data.write(str (self.new_speed) )
            data.close()

def train(env: str, steps: int, name: str):
        
    callback = CallBack( name = name, success_threshold = .95 )
    
    base = int(500 *env.unwrapped.tol /2)
    
    n_steps = base *env.unwrapped.num_pred
    
    batch_size = 100
    
    print(f'epoch: {n_steps} steps')
    
    try:
        
        try:
            with open(f'{name}_final_speed','r') as data:
                spd = data.read()
                data.close()
            speed = float (spd)
            
        except:
            speed = 0.
        
        env.env.env.options['new speed'] = speed
        
        model = CustomPPO.load(f"{name}", env, verbose=1, n_steps=n_steps, batch_size = batch_size )
        
    except:
        
        model = CustomPPO("MlpPolicy", env, verbose=1, n_steps=n_steps, batch_size = batch_size )
    
    #model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps, batch_size = batch_size )
    
    model.learn(total_timesteps=steps,
                callback=callback
                )
    
    callback.save_data()
    model.save(f"{name}")
    print(f'Final Speed: {callback.new_speed}')
    return callback.new_speed


def test(name: str, config: dict = {}, num_games: int = 10, wait: float = 0.0,
         render_mode: str | None = "human",
         record: str | None = None,
         coop: bool = False,
         var: bool = False,
         num_pred: int = 2,
         speed: float = 10.0):

    model = PPO.load(f"{name}")
    env = gym.make("SeqPredPreyEnv-v0", render_mode=render_mode, record=record, name = name,
                   options = config, var=var, coop=coop, num_pred = num_pred)

    CONFIG = {'prey_max_speed': speed}
    Rews = 0
    REWS = 0
    goals = 0

    g = 0
    while g < num_games:

        obs, info = env.reset(options=CONFIG)
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


def eva(env: str, name: str, num_games: int = 100, speed: float = 10.0):

    model = PPO.load(f"ppo_{name}")

    CONFIG = {'prey_max_speed': speed}
    Rews = 0
    REWS = 0
    goals = 0

    g = 0
    while g < num_games:

        obs, info = env.reset(options=CONFIG)
        done = False

        while not done:

            action = model.predict(obs, deterministic=True)[0]
            obs, rew, term, trunc, info = env.step(action)
            Rews += rew

            if term or trunc:
                REWS += Rews
                if term:
                    goals += 1

                #print(f'game: {g}, reward: {Rews}, terminated: {term}')

                g += 1
                Rews = 0
                done = True

    print('Average Reward: ', REWS/num_games)
    print('Catch Rate: ', goals/num_games)
    # return env #this should return info for inspection

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the SeqPredPrey environment")
    
    # Add arguments for various configuration options
    parser.add_argument('--coop', type=str, default= 'True', help="if rews is global or single agent")
    parser.add_argument('--var', type=str, default= 'True', help="if agents vary or not")
    parser.add_argument('--idx', type=int, default= 1, help="run index")
    parser.add_argument('--see_ally', type=str, default= 'True', help="if agents see partners")
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    coop = args.coop
    var = args.var
    idx = args.idx
    see_ally = args.see_ally

    scn = "SeqPredPreyEnv-v0"
    
    M = 1_000_000

    steps = 100  *M
    
    #coop, var, idx, see_ally = 'False', 'False', 14, 'False'
    #
    
    if coop == 'True':
        c = 'c'
        coop = True
        
    else:
        c = 's'  
        coop = False
    
    if var == 'True':
        v = 'v'
        min_mot = 0.9
        var = True                                                                
        
    else:
        v = 's'
        min_mot = 1.
        var = False
    
    if see_ally == 'False':
        name = f'{c}{v}b{idx}'
        sa = False
    else:
        name = f'{c}{v}{idx}'
        sa = True
        
    config = {'prey_max_speed': 0.,
             'min_motivation': min_mot,
             'tol': 20,
             'name': name,
             }
    
    print(name)
    
    env = gym.make(scn,
                   num_pred = 2,
                   coop=coop,
                   var = var,
                   name = name,
                   see_ally = sa,
                   options=config,
                   )
    
    #speed = 12
    speed = train(env=env, steps=steps, name=name)
    print(name)
    #eva(env=env, speed=speed, name=name)
    
    """
    test(name = name, var=var, coop=coop,
         speed = 10,
         #wait = 0.001,
         num_games = 100,
         )
    #"""

    env.close()

