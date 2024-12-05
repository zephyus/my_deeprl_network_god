"""
IA2C and MA2C algorithms
@author: Tianshu Chu
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, NCMultiAgentPolicy,
                             CommNetMultiAgentPolicy, DIALMultiAgentPolicy, NCLMMultiAgentPolicy)
import logging
import numpy as np


class IA2C:
    """
    The basic IA2C implementation with decentralized actor and centralized critic,
    limited to neighborhood area only.
    """
    # Decentralized: each agent decides its action based on its local observation
    # Critics are centralized but limited to a neighborhood area; each agent's critic can access information from neighboring agents to estimate the value function.

    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=True):
        self.name = 'ia2c'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, naction, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(
                ob[i], naction[i], action[i], reward, value[i], done)
    # 重點看一下backward他到底要看哪些東西

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(
                Rends[i], dt)
            if i == 0:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent):
            cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    def load(self, model_dir, global_step=None, train_mode=True):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        # 基本上所有的Agents自己都有一個action space, 意指所有的actions都在這個集合裡面，假如size都一樣，代表全部都是一樣的action
        # 這樣所有agents就可以分享同一組參數，加快學習。
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have varient dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            # 如果不一樣代表不是完全一致，全部的action dimension跟state dimension都設成最高的那個
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        # 這會在這會在init_policy中產生差異，基本上如果是identical，就會全員跑進一個比較簡單的LSTM中，如果比較難，則每一個agent
        # 會被個別訓練，等等會看到
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        # 下面這兩個好像是預處理等等reward要用的東西，兩個都存在model_config裡面是一個key
        # 反正就是超參數 哈哈
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        # 下面三個很直觀
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        # init torch
        print(torch.version.cuda)
        print(type(torch))
        # if(use_gpu):
        #    print("USEGPU")
        # if(torch.cuda.is_available()):
        #    print("CUDAISAVAIL")
        if use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.device = torch.device("cuda:0")
            logging.info('Use gpu for pytorch...')
        else:
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            self.device = torch.device("cpu")
            logging.info('Use cpu for pytorch...')
        # 先叫出一個policy，然後把它丟進GPU or CPU去跑 🆒
        self.policy = self._init_policy()
        self.policy.to(self.device)

        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            # 初始化training要的東西
            self._init_train(model_config, distance_mask, coop_gamma)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            if self.identical_agent:
                # 很簡單，policy全都一樣就是直接丟進去
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                # 不一樣就麻煩，我們要先知道na_dim_ls是啥子
                # na_dim_ls是一個list (程式裡面ls結尾就是list)，代表第i個鄰居他的action dimension是多少
                na_dim_ls = []
                # 下面這個np.where會從neighbor中找出mask是1的，然後自己組成一個新的tuple
                # 然後我們要再從tuple裡面抓出第一個，就可以抓到鄰居。
                # e.g. Suppose self.neighbor_mask[i] = [0, 1, 0, 1, 0]
                # np.where(self.neighbor_mask[i] == 1)  # Returns (array([1, 3]),)
                # np.where(self.neighbor_mask[i] == 1)[0]  # Returns array([1, 3])
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    # 這個迴圈就會把mask==1的鄰居的動作的次元load進na_dim_ls這個list中
                    na_dim_ls.append(self.n_a_ls[j])
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(
                                              i),
                                          na_dim_ls=na_dim_ls, identical=False)  # 這邊就會多一個na_dim_ls參數才可以確保lstm知道鄰居的動作次元
                # local_policy.to(self.device)
            policy.append(local_policy)  # 然後local的load進global的大list中
        return nn.ModuleList(policy)  # 模組化

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(
                self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # coopgamma == 0 -> 注重當前利益
        # coopgamma == 1 -> 注重未來利益
        # distance_mask 存的是每一個agent跟當前agent的距離
        # init lr(learning rate我不知道他爲什麼要簡寫好討厭) scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init,
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(
                gamma, coop_gamma, distance_mask[i]))

    def _update_lr(self):
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0,  use_gpu=True):
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, p, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.identical_agent:
            self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                             reward, value, done)
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(
            Rends, dt)
        self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                             summary_writer=summary_writer, global_step=global_step)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, ps, actions=None, out_type='p'):
        if self.identical_agent:
            return self.policy.forward(np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(
            gamma, coop_gamma, distance_mask)


class MA2C_NCLM(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, groups=0, seed=0, use_gpu=True):
        self.name = 'ma2c_nclm'
        self.groups = groups
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return NCLMMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm, groups=self.groups)
        else:
            return NCLMMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, groups=self.groups, identical=False)
