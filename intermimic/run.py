# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#############################################################################################################################A
import shutil
import subprocess
import sys
import yaml
from datetime import datetime
#############################################################################################################################A

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np

import wandb
from easydict import EasyDict

from learning import intermimic_agent
from learning import intermimic_players
from learning import intermimic_models
from learning import intermimic_network_builder


args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info
            # self.full_state["obs"]: tensor, not dictionary

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)

    runner.algo_factory.register_builder('intermimic', lambda **kwargs : intermimic_agent.InterMimicAgent(**kwargs))
    runner.algo_factory.register_builder('omnigrasp', lambda **kwargs : im_amp_agent.OmnigraspAmpAgent(**kwargs))

    runner.player_factory.register_builder('intermimic', lambda **kwargs : intermimic_players.InterMimicPlayerContinuous(**kwargs))
    runner.player_factory.register_builder('omnigrasp', lambda **kwargs : im_amp_players.OmnigraspAMPPlayerContinuous(**kwargs))

    runner.model_builder.model_factory.register_builder('intermimic', lambda network, **kwargs : intermimic_models.ModelInterMimicContinuous(network)) 
    runner.model_builder.model_factory.register_builder('omnigrasp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    
    runner.model_builder.network_factory.register_builder('intermimic', lambda **kwargs : intermimic_network_builder.InterMimicBuilder())
    runner.model_builder.network_factory.register_builder('omnigrasp', lambda **kwargs : amp_network_omnigrasp_builder.AMPOmniGraspBuilder())

    return runner


#############################################################################################################################A
def resolve_experiment_dir(cfg_train, add_timestamp=True):
    """Pin down <train_dir>/<experiment_name> and make rl_games agree with it.

    Left alone, rl_games derives the folder name itself (see A2CBase.__init__ in
    a2c_common.py) and never shares it, so we would snapshot into a different folder
    than the one holding the checkpoints. Deciding the name here and writing it back
    as `full_experiment_name` -- which rl_games uses verbatim -- keeps both in one place.
    """
    config = cfg_train['params']['config']
    experiment_name = config['name']
    if add_timestamp:
        experiment_name += datetime.now().strftime('_%d-%H-%M-%S')
    config['full_experiment_name'] = experiment_name
    return os.path.join(config.get('train_dir', 'runs'), experiment_name)


def save_run_artifacts(args, cfg_train, experiment_dir):
    """Snapshot everything needed to understand and rerun this experiment."""
    source_dir = os.path.join(experiment_dir, 'source')
    os.makedirs(source_dir, exist_ok=True)

    # 1. Freeze the configs. The originals under intermimic/data/cfg keep changing
    # across experiments, so we remember where each copy landed and point the
    # generated commands at those copies instead.
    test_cfg_env = args.cfg_env.replace('_train.yaml', '_test.yaml') if args.cfg_env else ''
    snapshot = {}
    for key, path in (('cfg_env', args.cfg_env),
                      ('cfg_train', args.cfg_train),
                      ('cfg_env_test', test_cfg_env)):
        if path and os.path.isfile(path):
            dst = os.path.join(source_dir, os.path.basename(path))
            shutil.copy2(path, dst)
            snapshot[key] = os.path.relpath(dst)

    # 2. The launching shell script, if we were started from one. bash execs python
    # as a child, so the script path only shows up in the parent's cmdline.
    try:
        import psutil
        for token in psutil.Process(os.getppid()).cmdline():
            if token.endswith('.sh') and os.path.isfile(token):
                shutil.copy2(token, os.path.join(source_dir, os.path.basename(token)))
                break
    except Exception:
        pass

    # 3. The resolved train config, capturing CLI overrides the copies above don't show.
    with open(os.path.join(source_dir, 'resolved_train_cfg.yaml'), 'w') as f:
        yaml.safe_dump(cfg_train, f, default_flow_style=False, sort_keys=False)

    # Keep each "--flag value" pair on one line so the written commands stay readable.
    argv, grouped, i = sys.argv[1:], [], 0
    while i < len(argv):
        if argv[i].startswith('-') and i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            grouped.append(f'{argv[i]} {argv[i + 1]}')
            i += 2
        else:
            grouped.append(argv[i])
            i += 1

    script = os.path.relpath(sys.argv[0])

    # 4. The exact command that launched this run -- a record of what was run.
    with open(os.path.join(source_dir, 'train_command.sh'), 'w') as f:
        f.write('#!/bin/bash\n# Auto-generated: the exact command that launched this run.\n')
        f.write('# References the ORIGINAL configs, which may have changed since.\n')
        f.write('# To rerun against this run\'s frozen configs, use reproduce_command.sh.\n')
        f.write('wandb online\n')
        f.write(' \\\n'.join([f'python {script}'] + grouped) + '\n')

    # 5. The same command against the frozen configs, so a rerun is unaffected by
    # later edits to the originals.
    repro = []
    for g in grouped:
        flag = g.split(' ')[0]
        if flag in ('--cfg_env', '--cfg_train') and flag[2:] in snapshot:
            repro.append(f'{flag} {snapshot[flag[2:]]}')
        else:
            repro.append(g)
    with open(os.path.join(source_dir, 'reproduce_command.sh'), 'w') as f:
        f.write('#!/bin/bash\n# Auto-generated: rerun this experiment against the configs\n')
        f.write('# frozen in this folder, not the originals under intermimic/data/cfg.\n')
        f.write('# NOTE: writes to a NEW timestamped folder; it will not overwrite this run.\n')
        f.write('wandb online\n')
        f.write(' \\\n'.join([f'python {script}'] + repro) + '\n')

    # 6. A test command wired to this run's own checkpoints.
    drop = ('--cfg_env', '--experiment', '--minibatch_size', '--num_envs',
            '--resume_from', '--max_iterations', '--horizon_length', '--headless')
    test_args = []
    for g in grouped:
        flag = g.split(' ')[0]
        if flag in drop:
            continue
        if flag == '--cfg_train' and 'cfg_train' in snapshot:
            g = f"--cfg_train {snapshot['cfg_train']}"
        test_args.append(g)

    nn_dir = os.path.join(experiment_dir, 'nn')
    ckpt = os.path.relpath(
        os.path.join(nn_dir, f"{cfg_train['params']['config']['name']}_latest.pth"))
    with open(os.path.join(source_dir, 'test_command.sh'), 'w') as f:
        f.write('#!/bin/bash\n# Auto-generated: swap the checkpoint below to test another epoch.\n')
        f.write(f'# Available checkpoints: {os.path.relpath(nn_dir)}\n')
        f.write('wandb offline\n')
        f.write(' \\\n'.join(
            [f'python {script}'] + test_args
            + [f"--cfg_env {snapshot.get('cfg_env_test', test_cfg_env)}",
               '--test', f'--checkpoint {ckpt}', '--num_envs 1', '--test_no_reset']) + '\n')

    # 7. Git provenance: the commit plus any uncommitted edits, so a saved run can be
    # traced back to the code that produced it.
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(['git', 'diff', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode()
        with open(os.path.join(source_dir, 'git_info.txt'), 'w') as f:
            f.write(f'commit: {sha}\n\n=== uncommitted diff ===\n{diff}')
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print(f'[run] Saved run artifacts to {source_dir}')

#############################################################################################################################A
def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    cfg = EasyDict(cfg)

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    if args.robot_type:
        cfg['env']['robotType'] = args.robot_type

    cfg['env']['testNoReset'] = args.test_no_reset

    # allow overriding resume_from via CLI
    if hasattr(args, 'resume_from') and (args.resume_from is not None) and (args.resume_from != 'None'):
        cfg_train['params']['config']['resume_from'] = args.resume_from

    if args.play_dataset:
        cfg['env']['playdataset'] = True

    if args.projtype:
        cfg['env']['projtype'] = args.projtype

    if args.cg1 != -1.:
        cfg['env']['rewardWeights']['cg1'] = args.cg1

    if args.cg2 != -1.:
        cfg['env']['rewardWeights']['cg2'] = args.cg2

    if args.ig != -1.:
        cfg['env']['rewardWeights']['ig'] = args.ig

    if args.op != -1.:
        cfg['env']['rewardWeights']['op'] = args.op

    if args.save_images:
        cfg['env']['saveImages'] = True
    
    if args.init_vel:
        cfg['env']['initVel'] = True

    if args.frames_scale!= 0.:
        cfg['env']['dataFramesScale'] = args.frames_scale

    if args.ball_size!= 0.:
        cfg['env']['ballSize'] = args.ball_size
    
    if args.collision != "None":
        # Parse collision argument: "all", "0", "[0,2]", etc.
        collision_value = args.collision
        if collision_value == "all":
            cfg['env']['humanObjectCollision'] = "all"
        elif collision_value.startswith('[') and collision_value.endswith(']'):
            # Parse list format: "[0,2]" -> [0, 2]
            import json
            cfg['env']['humanObjectCollision'] = json.loads(collision_value)
        else:
            # Try to parse as integer
            try:
                cfg['env']['humanObjectCollision'] = int(collision_value)
            except ValueError:
                cfg['env']['humanObjectCollision'] = collision_value

    if args.grasp_window_before != -1:
        cfg['env']['graspWindowBefore'] = args.grasp_window_before

    if args.grasp_window_after != -1:
        cfg['env']['graspWindowAfter'] = args.grasp_window_after

    if args.wrist_pos_reset_threshold1 != -1.0:
        cfg['env']['wristPosResetThreshold1'] = args.wrist_pos_reset_threshold1

    if args.wrist_rot_reset_threshold1 != -1.0:
        cfg['env']['wristRotResetThreshold1'] = args.wrist_rot_reset_threshold1

    if args.wrist_pos_reset_threshold2 != -1.0:
        cfg['env']['wristPosResetThreshold2'] = args.wrist_pos_reset_threshold2

    if args.wrist_rot_reset_threshold2 != -1.0:
        cfg['env']['wristRotResetThreshold2'] = args.wrist_rot_reset_threshold2

    if args.finger_window_before != -1:
        cfg['env']['fingerWindowBefore'] = args.finger_window_before

    if args.finger_window_after != -1:
        cfg['env']['fingerWindowAfter'] = args.finger_window_after

    if args.reset_transition_frame != -1:
        cfg['env']['resetTransitionFrame'] = args.reset_transition_frame

    if args.reset_pd_action_offset == True:
        cfg['env']['reset_pd_action_offset'] = args.reset_pd_action_offset

    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path

    # Give each training run its own folder and snapshot the run's inputs into it
    if not args.test:
        experiment_dir = resolve_experiment_dir(cfg_train)
        os.makedirs(experiment_dir, exist_ok=True)
        save_run_artifacts(args, cfg_train, experiment_dir)

    # Initialize wandb (resume if wandb_id is provided)
    wandb_kwargs = {
        "project": "intermimic",
        "name": cfg_train['params']['config']['name'],
        "config": {
            "task": args.task,
            "cfg_env": args.cfg_env,
            "cfg_train": args.cfg_train,
            "num_envs": args.num_envs,
            'seed': cfg_train['params']['seed'],
            **cfg,
            **cfg_train
        },
        "id":args.wandb_id,
        "resume":"allow"
    }

    wandb.init(**wandb_kwargs)
    
    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return

if __name__ == '__main__':
    main()
