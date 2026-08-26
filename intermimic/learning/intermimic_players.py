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

import torch, time
import sys
import select
import termios
import tty

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import learning.common_player as common_player

class InterMimicPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', False)
        self.is_paused = False
        self.old_terminal_settings = None

        super().__init__(config)
        return

    def _setup_keyboard_input(self):
        """Setup keyboard input for pause functionality"""
        # Check if we're in a debugger (pdb, ipdb, etc.)
        import inspect
        for frame in inspect.stack():
            if frame.filename.endswith('pdb.py') or frame.filename.endswith('bdb.py'):
                print("Debugger detected - pause functionality disabled")
                self.old_terminal_settings = None
                return

        try:
            self.old_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except:
            print("Warning: Keyboard input not available in this environment")
            self.old_terminal_settings = None

    def _cleanup_keyboard_input(self):
        """Restore terminal settings"""
        if self.old_terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            except:
                pass

    def _handle_keyboard_input(self, current_frame=None):
        """Handle keyboard input for pause/resume"""
        if self.old_terminal_settings is None:
            return

        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                if key == ' ':
                    if not self.is_paused:
                        self.is_paused = True
                        frame_info = f" at frame {current_frame}" if current_frame is not None else ""
                        print(f"\n⏸️  PAUSED{frame_info} (Press SPACE to resume)")
                    else:
                        self.is_paused = False
                        print("\n▶️  RESUMED")
        except:
            pass

    def run(self):
        # Setup keyboard input
        # self._setup_keyboard_input()
        print("\nControls: SPACE = Pause/Resume")

        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        n_games = n_games * n_game_life
        n_games = 1
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        # Success rate tracking
        success_count = 0
        fail_count = 0
        num_test_episodes = getattr(self.env.task, 'num_test_episodes', -1)
        num_test_episodes = 10000

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        try:
            for _ in range(n_games):
                if games_played >= n_games:
                    break

                obs_dict = self.env_reset()
                batch_size = 1
                batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

                if need_init_rnn:
                    self.init_rnn()
                    need_init_rnn = False

                cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

                # Object error metric buffers
                obj_pos_error_sum = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                obj_rot_error_sum = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                error_steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                episode_obj_pos_errors = []
                episode_obj_rot_errors = []

                done_indices = []

                if self.env.task.play_dataset:
                    # play dataset
                    while True:
                        for t in range(self.env.task.max_episode_length.max()):
                            # Handle keyboard input for pause
                            # self._handle_keyboard_input(current_frame=t)

                            # Only step if not paused
                            if not self.is_paused:
                                self.env.task.play_dataset_step(t)
                            else:
                                # When paused, still render the current frame
                                if render:
                                    self.env.render(mode='human')
                                time.sleep(0.01)  # Small sleep to reduce CPU usage while paused
                else:
                    # inference
                    for n in range(self.max_steps):
                        # Get actual frame number from environment (first env)
                        current_frame = self.env.task.progress_buf[0].item() if hasattr(self.env.task, 'progress_buf') else n

                        # Handle keyboard input for pause
                        # self._handle_keyboard_input(current_frame=current_frame)

                        # Only step if not paused
                        # if not self.is_paused:
                        if not self.env.task.paused or self.env.task.step_once:
                            obs_dict = self.env_reset(done_indices)

                            if has_masks:
                                masks = self.env.get_action_mask()
                                action = self.get_masked_action(obs_dict, masks, is_determenistic)
                            else:
                                action = self.get_action(obs_dict, is_determenistic)
                            obs_dict, r, done, info =  self.env_step(self.env, action)
                            cr += r
                            steps += 1

                            self._post_step(info)

                            # Accumulate object error metrics
                            obj_pos_err = info.get('obj_pos_error', None)
                            if obj_pos_err is not None:
                                obj_pos_error_sum += obj_pos_err
                                obj_rot_error_sum += info.get('obj_rot_error', torch.zeros_like(obj_pos_err))
                                error_steps += 1

                            all_done_indices = done.nonzero(as_tuple=False)
                            done_indices = all_done_indices[::self.num_agents]
                            done_count = len(done_indices)
                            games_played += done_count

                            if done_count > 0:
                                if self.is_rnn:
                                    for s in self.states:
                                        s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                                cur_rewards = cr[done_indices].sum().item()
                                cur_steps = steps[done_indices].sum().item()

                                cr = cr * (1.0 - done.float())
                                steps = steps * (1.0 - done.float())
                                sum_rewards += cur_rewards
                                sum_steps += cur_steps

                                # Collect per-episode object error metrics
                                for idx in done_indices:
                                    i = idx.item()
                                    if error_steps[i] > 0:
                                        episode_obj_pos_errors.append((obj_pos_error_sum[i] / error_steps[i]).item())
                                        episode_obj_rot_errors.append((obj_rot_error_sum[i] / error_steps[i]).item())
                                    obj_pos_error_sum[i] = 0
                                    obj_rot_error_sum[i] = 0
                                    error_steps[i] = 0

                                # Success rate tracking
                                terminate_buf = info.get('terminate', None)
                                if terminate_buf is not None:
                                    for idx in done_indices:
                                        i = idx.item()
                                        if terminate_buf[i].item() == 0:
                                            success_count += 1
                                        else:
                                            fail_count += 1
                                    total = success_count + fail_count
                                    obj_pos_msg = f" | obj_pos_err: {sum(episode_obj_pos_errors)/len(episode_obj_pos_errors):.4f}m" if episode_obj_pos_errors else ""
                                    obj_rot_msg = f" | obj_rot_err: {sum(episode_obj_rot_errors)/len(episode_obj_rot_errors):.4f}rad" if episode_obj_rot_errors else ""
                                    print(f"[Eval] {total} episodes | Success: {success_count} | Fail: {fail_count} | Rate: {success_count/total*100:.1f}%{obj_pos_msg}{obj_rot_msg}")

                                    # Stop if num_test_episodes reached
                                    if num_test_episodes > 0 and total >= num_test_episodes:
                                        break

                            done_indices = done_indices[:, 0]
                        else:
                            # When paused, still render
                            if render:
                                self.env.task.render()
                            time.sleep(0.01)  # Small sleep to reduce CPU usage while paused
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self._cleanup_keyboard_input()

        # Print final success rate
        total = success_count + fail_count
        if total > 0:
            print(f"\n========== Final Evaluation Results ==========")
            print(f"Total episodes  : {total}")
            print(f"Success         : {success_count}")
            print(f"Fail            : {fail_count}")
            print(f"Success Rate    : {success_count / total * 100:.2f}%")
            if episode_obj_pos_errors:
                print(f"Avg obj_pos_err : {sum(episode_obj_pos_errors)/len(episode_obj_pos_errors):.4f} m")
            if episode_obj_rot_errors:
                print(f"Avg obj_rot_err : {sum(episode_obj_rot_errors)/len(episode_obj_rot_errors):.4f} rad")
            print(f"================================================")

        return
    
    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        return