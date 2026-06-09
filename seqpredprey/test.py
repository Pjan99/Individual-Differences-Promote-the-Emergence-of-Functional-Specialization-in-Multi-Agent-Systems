# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:51:38 2025

@author: pjan99
"""
# ===== NumPy 2 -> 1.26 compatibility shim (keep when loading pickles saved under NumPy>=2) =====
import sys, types, numpy as _np
if '_core' not in _np.__dict__:
    np_core_mod = types.ModuleType('numpy._core')
    sys.modules['numpy._core'] = np_core_mod
    for s in ['numeric', 'fromnumeric', 'shape_base', 'multiarray', 'umath', 'arrayprint', 'getlimits']:
        try:
            sys.modules[f'numpy._core.{s}'] = getattr(_np.core, s)
        except Exception:
            pass
# -----------------------------------------------------------------------------------------------

# === Silence only the two SB3 schedule warnings (safe for evaluation) ===
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Could not deserialize object (clip_range|lr_schedule)\.",
    module=r"stable_baselines3\.common\.save_util",
)

import os
import math
import time
import json
import numpy as np
import pandas as pd
import gymnasium as gym

import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import seqpredprey

from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Configuration you can tweak ----------------

# Episode counts / speeds
num_games = 100                    
speeds = list(range(10, 21, 1))    # 10..20 (inclusive)

# Run which blocks? (set to False to skip)
RUN_SWEEP_NATIVE     = True
RUN_SWEEP_SOFT_FLIP  = False
RUN_PREF_NATIVE      = True
RUN_PREF_SOFT_FLIP   = True

# Catch radius (prey+pred radii). Adjust if your env changes these.
CATCH_RADIUS_SUM = 8.5 + 8.5

# Seeds per condition (normal vs blind)
SEEDS_NORMAL = range(1, 31, 1)  # 1..30
SEEDS_BLIND  = range(1, 11, 1)   # 1..10

# Conditions you want to evaluate (normal + blind)
CONDITIONS = ['cs', 'cv', 'ss', 'sv', 
              'csb', 'cvb', 'ssb', 'svb'
              ]

#NORMAL_CONDS = {'cs','cv','ss','sv'}
#BLIND_CONDS  = {'csb','cvb','ssb','svb'}

# ------------------------------------------------------------


# --------------- Helpers -----------------
def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt((p*(1-p)/n) + z**2/(4*n**2)) / denom
    return center - half, center + half

def cond_parse(cond_full):
    """
    cond_full: e.g., 'cs', 'cv', 'ss', 'sv', 'csb', 'cvb', 'ssb', 'svb'
    Returns:
      base_cond: 'cs'/'cv'/'ss'/'sv' (2 letters)
      blind: int 0/1
      coop: 1 if first letter 'c' else 0
      var:  1 if second letter 'v' else 0
    """
    blind = 1 if cond_full.endswith('b') else 0
    base = cond_full[:2]
    coop = 1 if base[0] == 'c' else 0
    var  = 1 if base[1] == 'v' else 0
    return base, blind, coop, var

def cond_to_motivation_bounds(var_flag):
    """Training-time bounds used in your original code."""
    min_mot = 0.9 if var_flag else 1.0
    # The env reset uses: max_motivation = 1. + (1 - min_motivation)
    # So when min_mot=1.0 => max_mot=1.0 (homogeneous); when min_mot=0.9 => max_mot=1.1 (varied)
    return min_mot

def seed_everything(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)

def distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

# 16 slices of 22.5° each in the PREY-centric frame.
# predator_angle_relative_to_prey() returns ang_rel where:
#   0 rad = predator in front of prey
#   +pi/2 = predator to prey's LEFT (CCW)
#   -pi/2 = predator to prey's RIGHT (CW)
#
# Your requested bins go 0..360 increasing toward the RIGHT side first,
# so we convert to CLOCKWISE degrees via: deg_cw = (-ang_rel) in degrees.

SECTOR_16_NAMES = (
    "front_centre_right",   # 0   .. 22.5
    "front_right",          # 22.5.. 45
    "right_left",           # 45  .. 67.5
    "right_centre_left",    # 67.5.. 90
    "right_centre_right",   # 90  .. 112.5
    "right_right",          # 112.5..135
    "back_left",            # 135 ..157.5
    "back_centre_left",     # 157.5..180
    "back_centre_right",    # 180 ..202.5
    "back_right",           # 202.5..225
    "left_left",            # 225 ..247.5
    "left_centre_left",     # 247.5..270
    "left_centre_right",    # 270 ..292.5
    "left_right",           # 292.5..315
    "front_left",           # 315 ..337.5
    "front_centre_left",    # 337.5..360
)

def angle_to_sector_16(angle_rad: float) -> str:
    """
    Map prey-frame relative angle (radians) to one of 16 sectors (22.5° each),
    following the user's clockwise 0..360° definitions.

    - 0° is "front" of prey
    - 0..360 increases toward prey's RIGHT side first
    """
    a = float(angle_rad)

    # normalize to [-pi, pi)
    a = (a + math.pi) % (2 * math.pi) - math.pi

    # convert to clockwise degrees in [0, 360)
    deg_cw = (-a) * (180.0 / math.pi)
    deg_cw = deg_cw % 360.0

    bin_width = 22.5
    idx = int(deg_cw // bin_width)  # 0..15

    return SECTOR_16_NAMES[idx]

def predator_angle_relative_to_prey(envu, predator_id: str, prey_id: str = None) -> float:
    """
    Compute angle of predator around prey, in prey's reference frame.

    Returns: angle in radians in [-pi, pi).

    - predator_id like 'pred0'
    - prey_id optional; if None and multiple prey exist, uses the closest prey
    """
    # Choose prey
    if prey_id is None:
        # if there's only one prey, this picks it; else picks closest
        prey_keys = list(envu.preys.keys())
        if len(prey_keys) == 0:
            raise RuntimeError("No prey found in envu.preys")

        if len(prey_keys) == 1:
            prey_id = prey_keys[0]
        else:
            px = envu.agents[predator_id].x_pos
            py = envu.agents[predator_id].y_pos
            best = None
            best_d2 = None
            for k in prey_keys:
                dx = px - envu.preys[k].x_pos
                dy = py - envu.preys[k].y_pos
                d2 = dx*dx + dy*dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = k
            prey_id = best

    # Positions
    pred = envu.agents[predator_id]
    prey = envu.preys[prey_id]

    dx = pred.x_pos - prey.x_pos
    dy = pred.y_pos - prey.y_pos

    # World-frame angle from prey -> predator
    ang_world = math.atan2(dy, dx)

    # Convert to prey frame: 0 means "in front of prey"
    ang_rel = ang_world - prey.ori

    # normalize to [-pi, pi)
    ang_rel = (ang_rel + math.pi) % (2 * math.pi) - math.pi
    return ang_rel

def predator_sector_relative_to_prey(envu, predator_id: str, prey_id: str = None) -> str:
    """
    Convenience: env state -> prey-frame relative angle -> 16-sector (22.5°) label.
    """
    ang = predator_angle_relative_to_prey(envu, predator_id, prey_id=prey_id)
    return angle_to_sector_16(ang)

def load_preferred_speed(name, base_dir):
    """
    Reads the validated preferred prey speed for this trained controller.
    """
    path = os.path.join(base_dir, f"{name}_preferred_speed.txt")
    with open(path, "r") as f:
        return float(f.read().strip())

def seeds_for_condition(cond_full):
    """Use 1..9 for normal, 1..3 for blind conditions."""
    return SEEDS_BLIND if cond_full.endswith('b') else SEEDS_NORMAL

def build_env(name, base_cond, coop, var, blind, soft_flip, prey_max_speed, min_mot_base):
    """
    Build an env with options set according to training var flag and soft_flip.
    - If var==True and soft_flip==True => env forces homogeneous (equal-speed) by design.
    - If var==False and soft_flip==True => we emulate 'varied' eval by lowering min_mot to 0.9.
      (Because your env only *forces* homogeneous when (var==False and flip==False) or (var==True and flip==True)).
    """
    # Decide the evaluation 'min_motivation' based on soft_flip logic:
    if soft_flip and (var == 0):
        # equal-trained -> evaluate as 'varied': use 0.9 so env will create spread (since flip==True doesn't force homo)
        min_mot_eval = 0.9
    else:
        # native behavior, i.e., what you used at training time
        min_mot_eval = min_mot_base

    options = {
        'prey_max_speed': prey_max_speed,
        'min_motivation': float(min_mot_eval),
        'tol': 20,
        'name': name,
        'soft_flip': bool(soft_flip),
        # leave 'max_motivation' to env default (1 + (1 - min_motivation))
    }

    env = gym.make(
        "SeqPredPreyEnv-v0",
        name=name,
        options=options,
        var=bool(var),
        coop=bool(coop),
        see_ally=(blind == 0),   # blind==1 => see_ally=False
        num_pred=2
    )
    return env

def eval_one_setting(model, env, speeds_or_single, num_games, meta_cols, collect_pref=False, pref_sector_rows_out=None):
    """
    Run either a sweep (list of speeds) or a single preferred speed (float).
    Returns:
      episodes_rows: list of dict (per episode)
      summary_rows: list of dict (per condition/seed/speed)
      pref_episodes_rows / pref_summary_rows populated when collect_pref=True
    """
    episodes_rows = []
    summary_rows  = []

    pref_episodes_rows = []
    pref_summary_rows  = []

    envu = env.unwrapped
    is_sweep = isinstance(speeds_or_single, (list, tuple))
    speeds_iter = speeds_or_single if is_sweep else [speeds_or_single]

    # For preferred-speed per-predator aggregation
    pref_catcher_totals = None
    pref_ties_total = 0
    pref_catches_total = 0

    for spd in speeds_iter:
        # Update prey speed for each block of episodes
        env.unwrapped.options['prey_max_speed'] = spd

        ep_returns = []
        catches = 0

        # For per-speed per-episode catcher/tie flags
        for g in range(num_games):
            env.unwrapped.options['seed'] = g
            obs, info = env.reset(options=env.unwrapped.options)
            ep_return = 0.0
            
            # For 30° bins, we use the shared label list + 'unknown' safety bucket.
            sector_names = SECTOR_16_NAMES
            sector_counts = None
            sector_steps = None
            if collect_pref and (pref_sector_rows_out is not None):
                sector_counts = {pid: {s: 0 for s in sector_names} for pid in envu.agents.keys()}
                sector_steps  = {pid: 0 for pid in envu.agents.keys()}

            terminated = truncated = False
            while not (terminated or truncated):
                actor = f'pred{envu.agent_iter}'  # current acting predator (string)
                
                if collect_pref and (pref_sector_rows_out is not None):
                    sector = predator_sector_relative_to_prey(envu, actor)
                    sector_counts[actor][sector] += 1
                    sector_steps[actor] += 1
                        
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = env.step(action)
                ep_return += float(rew)

            # per-episode catcher/tie
            tie_flag = 0
            catcher_label = actor
            ai = envu.agent_iter
            if terminated:
                catches += 1
                prey = envu.preys['prey0']
                if int(actor[-1]) != len(envu.agents) -1:
                    for _ in range(ai, len(envu.agents),1 ):
                        
                        p = envu.agents[f'pred{envu.agent_iter}']
                        
                        #print('checking', actor)
                        action = model.predict(obs, deterministic=True)[0]
                        
                        obs, ghost_rew, ghost_term, ghost_trunc, ghost_info = env.step(action)
                        
                        ai = envu.agent_iter

                        if distance(p.x_pos, p.y_pos, prey.x_pos, prey.y_pos) <= CATCH_RADIUS_SUM:
                            tie_flag = 1
                            catcher_label = 'none'

            ep_row = dict(meta_cols)
            ep_row.update({
                'speed': float(spd),
                'episode': g,
                'return': ep_return,
                'caught': bool(terminated),
                'tie': int(tie_flag),
                'catcher': catcher_label,
            })
            episodes_rows.append(ep_row)

            # If collecting preferred-only aggregates:
            if collect_pref:
                if pref_catcher_totals is None:
                    pref_catcher_totals = {pid: 0 for pid in envu.agents.keys()}  # e.g. pred0, pred1
                if terminated:
                    pref_catches_total += 1
                    if tie_flag == 1:
                        pref_ties_total += 1
                    elif catcher_label in pref_catcher_totals:
                        pref_catcher_totals[catcher_label] += 1
                pref_episodes_rows.append(ep_row)
            
                #if collecting sector profile on preferred-only
                if (pref_sector_rows_out is not None):
                    sec_row = dict(meta_cols)
                    sec_row.update({
                        'speed': float(spd),
                        'episode': g,
                    })
                    for pid in envu.agents.keys():
                        denom = sector_steps.get(pid, 0)
                        for s in SECTOR_16_NAMES:
                            key = f"{pid}_{s}_freq"
                            if denom > 0:
                                sec_row[key] = sector_counts[pid][s] / denom
                            else:
                                sec_row[key] = float('nan')
                    pref_sector_rows_out.append(sec_row)
                
            ep_returns.append(ep_return)

        # Per-speed summary
        avg_ret = float(np.mean(ep_returns)) if ep_returns else float('nan')
        std_ret = float(np.std(ep_returns)) if ep_returns else float('nan')
        catch_rate = catches / num_games if num_games > 0 else float('nan')
        lo, hi = wilson_ci(catch_rate, num_games) if num_games > 0 else (float('nan'), float('nan'))

        sum_row = dict(meta_cols)
        sum_row.update({
            'speed': float(spd),
            'avg_return': avg_ret,
            'std_return': std_ret,
            'catch_rate': catch_rate,
            'catch_rate_lo': lo,
            'catch_rate_hi': hi,
            'games': num_games,
        })
        summary_rows.append(sum_row)

    # Preferred-speed aggregate (per predator + ties)
    if collect_pref:
        # defender: if no episodes were run, handle gracefully
        if pref_catcher_totals is None:
            pref_catcher_totals = {pid: 0 for pid in envu.agents.keys()}
        total_games = num_games * len(speeds_iter)
        avg_ret = float(np.mean([r['return'] for r in pref_episodes_rows])) if pref_episodes_rows else float('nan')
        std_ret = float(np.std([r['return'] for r in pref_episodes_rows])) if pref_episodes_rows else float('nan')
        catch_rate = pref_catches_total / total_games if total_games > 0 else float('nan')
        lo, hi = wilson_ci(catch_rate, total_games) if total_games > 0 else (float('nan'), float('nan'))

        pref_sum_row = dict(meta_cols)
        pref_sum_row.update({
            'avg_return': avg_ret,
            'std_return': std_ret,
            'catch_rate': catch_rate,
            'catch_rate_lo': lo,
            'catch_rate_hi': hi,
            'ties': pref_ties_total,
            'games': total_games,
        })
        # per-pred catch rates
        for pid, cnt in pref_catcher_totals.items():
            pref_sum_row[f'catch_rate_{pid}'] = cnt / total_games if total_games > 0 else float('nan')

        pref_summary_rows.append(pref_sum_row)

    return episodes_rows, summary_rows, pref_episodes_rows, pref_summary_rows


# ------------------ Main -------------------
def main():
    seed_everything(0)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    all_episodes = []
    all_summary  = []
    pref_episodes_all = []
    pref_summary_all  = []
    pref_sector_all = []

    for cond_full in CONDITIONS:
        base_cond, blind, coop, var = cond_parse(cond_full)
        min_mot_base = cond_to_motivation_bounds(var)

        # Seed loop depends on blind/normal
        for ns in seeds_for_condition(cond_full):
            name = f"{cond_full}{ns}"
            print(f"== {name} ==")

            # ---- Load model ----
            model = PPO.load(f"{name}_update")

            # ---------- Block A: SWEEP (native) ----------
            if RUN_SWEEP_NATIVE:
                env = build_env(
                    name=name, base_cond=base_cond, coop=coop, var=var, blind=blind,
                    soft_flip=False, prey_max_speed=0, min_mot_base=min_mot_base
                )
                meta = {
                    'condition_full': cond_full,
                    'condition': base_cond,      # 2-letter for analysis (cs/cv/ss/sv)
                    'blind': int(blind),
                    'agent': int(ns),
                    'soft_flip': 0,
                    'train_variant': int(var),   # trained with variability?
                    'eval_variant': int(var),    # native eval matches train
                    'run_type': 'sweep_native',
                }
                epi, summ, _, _ = eval_one_setting(model, env, speeds, num_games, meta, collect_pref=False)
                all_episodes.extend(epi)
                all_summary.extend(summ)
                env.close()

            # ---------- Block B: SWEEP (soft_flip) ----------
            # var==1 => becomes equal-speed at eval; var==0 => becomes varied at eval
            if RUN_SWEEP_SOFT_FLIP:
                env = build_env(
                    name=name, base_cond=base_cond, coop=coop, var=var, blind=blind,
                    soft_flip=True, prey_max_speed=0, min_mot_base=min_mot_base
                )
                eval_variant = 0 if var == 1 else 1
                meta = {
                    'condition_full': cond_full,
                    'condition': base_cond,
                    'blind': int(blind),
                    'agent': int(ns),
                    'soft_flip': 1,
                    'train_variant': int(var),
                    'eval_variant': int(eval_variant),
                    'run_type': 'sweep_softflip',
                }
                # NOTE:
                # soft_flip corresponds to the "Flipped" evaluation condition
                # described in the manuscript.
                #
                # Same-speed trained agents are evaluated with Varied speeds.
                # Varied-speed trained agents are evaluated with Same speeds
                # while retaining their original proprioceptive cue.

                epi, summ, _, _ = eval_one_setting(model, env, speeds, num_games, meta, collect_pref=False)
                all_episodes.extend(epi)
                all_summary.extend(summ)
                env.close()

            # ---------- Block C: Preferred speed (native) ----------
            if RUN_PREF_NATIVE:
                try:
                    preferred_speed = load_preferred_speed(name, SCRIPT_DIR)
                except FileNotFoundError:
                    print(f"[WARN] Missing preferred-speed file for {name}; skipping preferred-native.")
                else:
                    env = build_env(
                        name=name, base_cond=base_cond, coop=coop, var=var, blind=blind,
                        soft_flip=False, prey_max_speed=preferred_speed, min_mot_base=min_mot_base
                    )
                    meta = {
                        'condition_full': cond_full,
                        'condition': base_cond,
                        'blind': int(blind),
                        'agent': int(ns),
                        'soft_flip': 0,
                        'train_variant': int(var),
                        'eval_variant': int(var),
                        'run_type': 'preferred_native',
                        'preferred_speed': float(preferred_speed),
                    }
                    epi, _, pref_epi, pref_sum = eval_one_setting(
                        model, env, preferred_speed, num_games, meta, collect_pref=True,
                        pref_sector_rows_out=pref_sector_all
                    )
                    all_episodes.extend(epi)            # includes the preferred episodes too
                    pref_episodes_all.extend(pref_epi)
                    pref_summary_all.extend(pref_sum)
                    env.close()

            # ---------- Block D: Preferred speed (soft_flip) ----------
            if RUN_PREF_SOFT_FLIP and (blind == 0):
                try:
                    preferred_speed = load_preferred_speed(name, SCRIPT_DIR)
                except FileNotFoundError:
                    print(f"[WARN] Missing preferred-speed file for {name}; skipping preferred-softflip.")
                else:
                    env = build_env(
                        name=name, base_cond=base_cond, coop=coop, var=var, blind=blind,
                        soft_flip=True, prey_max_speed=preferred_speed, min_mot_base=min_mot_base
                    )
                    eval_variant = 0 if var == 1 else 1
                    meta = {
                        'condition_full': cond_full,
                        'condition': base_cond,
                        'blind': int(blind),
                        'agent': int(ns),
                        'soft_flip': 1,
                        'train_variant': int(var),
                        'eval_variant': int(eval_variant),
                        'run_type': 'preferred_softflip',
                        'preferred_speed': float(preferred_speed),
                    }
                    # NOTE:
                    # soft_flip corresponds to the "Flipped" evaluation condition
                    # described in the manuscript.
                    #
                    # Same-speed trained agents are evaluated with Varied speeds.
                    # Varied-speed trained agents are evaluated with Same speeds
                    # while retaining their original proprioceptive cue.

                    epi, _, pref_epi, pref_sum = eval_one_setting(
                        model, env, preferred_speed, num_games, meta, collect_pref=True,
                        pref_sector_rows_out=pref_sector_all
                    )
                    all_episodes.extend(epi)
                    pref_episodes_all.extend(pref_epi)
                    pref_summary_all.extend(pref_sum)
                    env.close()

    # --------- Save outputs ---------
    df_epi = pd.DataFrame(all_episodes)
    df_sum = pd.DataFrame(all_summary)
    df_pref_epi = pd.DataFrame(pref_episodes_all)
    df_pref_sum = pd.DataFrame(pref_summary_all)
    df_pref_sector = pd.DataFrame(pref_sector_all)

    df_epi.to_csv(
        os.path.join(RESULTS_DIR, "eval_results_episodes_all.csv"),
        index=False
    )
    
    df_sum.to_csv(
        os.path.join(RESULTS_DIR, "eval_results_summary_all.csv"),
        index=False
    )
    
    if not df_pref_epi.empty:
        df_pref_epi.to_csv(
            os.path.join(RESULTS_DIR, "eval_results_preferred_episodes_all.csv"),
            index=False
        )
    
    if not df_pref_sum.empty:
        df_pref_sum.to_csv(
            os.path.join(RESULTS_DIR, "eval_results_preferred_summary_all.csv"),
            index=False
        )
    
    if not df_pref_sector.empty:
        df_pref_sector.to_csv(
            os.path.join(RESULTS_DIR, "eval_results_preferred_sector_freq.csv"),
            index=False
        )

    # Quick console checks
    if not df_sum.empty:
        print("\n=== Sweep summary (mean by condition/agent/run_type) ===")
        print(
            df_sum.groupby(['condition', 'blind', 'agent', 'run_type'])[['avg_return', 'catch_rate']]
                 .mean()
                 .round(4)
        )
    if not df_pref_sum.empty:
        print("\n=== Preferred summary (per cond/agent/run_type) ===")
        keep = ['avg_return', 'catch_rate', 'ties'] + [c for c in df_pref_sum.columns if c.startswith('catch_rate_pred')]
        print(df_pref_sum[['condition','blind','agent','run_type'] + keep].round(4))

if __name__ == "__main__":
    main()
