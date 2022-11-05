# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.observation_space = np.zeros(self.num_obs)
        self.action_space = np.zeros(self.num_acts)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)
        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RaisimGymVecTorchEnv:
    def __init__(self, impl, cfg, normalize_ob=False, seed=0, normalize_rew=False, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._kinematic_observation = np.zeros([self.num_envs, 84], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.float32)
        self._time_limit_done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.observation_space = np.zeros(self.num_obs)
        self.action_space = np.zeros(self.num_acts)

        self._observation_torch = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=device)
        self._kinematic_observation_torch = torch.zeros(self.num_envs, 84, dtype=torch.float32, device=device)
        self._reward_torch = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._done_torch = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

        self.total_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

        self.num_states = self.wrapper.getStateDim()
        self._state = np.zeros([self.num_envs, self.num_states], dtype=np.float32)
        self._state_torch = torch.zeros(self.num_envs, self.num_states, dtype=torch.float32, device=device)

        # time limit
        self.t = torch.zeros(self.num_envs, dtype=torch.int32)
        self.time_limit = 150

        self.num_steps = np.zeros(self.num_envs, dtype=np.int32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action.to("cpu").numpy(), self._reward, self._done)
        self.wrapper.observe(self._observation)
        self._reward_torch[:] = torch.from_numpy(self._reward[:]).to(device)
        self._done_torch[:] = torch.from_numpy(self._done[:]).to(device)

        count = torch.bincount(self._done_torch)

        for termination_condition_idx in range(1, count.shape[0]):
            self._done_count[termination_condition_idx - 1] += count[termination_condition_idx]

        self.get_total_reward()
        self.num_steps += 1
        return self.observe(), self._reward_torch[:], self._done_torch[:], {}

    def set_reference(self, reference):
        self.wrapper.setReference(reference)

    def set_reference_velocity(self, reference_velocity):
        self.wrapper.setReferenceVelocity(reference_velocity)

    def get_reference(self):
        self.wrapper.getReference(self._kinematic_observation)
        self._kinematic_observation_torch[:, :] = torch.from_numpy(self._kinematic_observation[:, :]).to(device)
        return self._kinematic_observation_torch

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self):
        self.wrapper.observe(self._observation)
        self._observation_torch[:, :] = torch.from_numpy(self._observation[: ,:]).to(device)
        return self._observation_torch

    def reset(self):
        self._reward[:] = np.zeros(self.num_envs, dtype=np.float32)
        self._done_count = np.zeros(4)
        self.wrapper.reset()

    def done_reset(self, done):
        self.wrapper.done_reset(done)

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def reset_time_limit(self):
        self.wrapper.timeLimitReset(self._time_limit_done)
        return self._time_limit_done.copy()

    def get_total_reward(self):
        total_rewards = self.wrapper.getTotalRewards();
        self.total_rewards[:] = torch.from_numpy(total_rewards).to(device)
        # reward logging
        self.reward_info = self.wrapper.rewardInfo()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def setTask(self):
        self.wrapper.setEnvironmentTask()

    def get_state(self):
        self.wrapper.getState(self._state)
        self._state_torch[:, :] = torch.from_numpy(self._state[: ,:]).to(device)
        return self._state_torch[:, :]

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

