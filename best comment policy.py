import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn


class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a   #number of action
        self.n_s = n_s   #number of state(an agent can observe)
        self.n_step = n_step
        self.identical = identical

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    # init_actor_head，是為了初始化一個actor這個頭的初始狀態，最一開始會從hidden layer匯入進來，那hidden layer則就會跟action layer
    # 形成一個全連接層，
    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # hidden與action層連接
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')
    # critic head

    # initializes a neural network layer (critic head) for estimating the value function.
    # 作用就是  設定神經網路結構的參數（hidden layer 維度等）。
    # 輸出	沒有直接輸出，只初始化神經網路層。
    # 作用	創建 critic_head（模型結構）
    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)   # n_n 就是number of neighbors
        if n_n:
            # 這邊要看有沒有identical，如果大家都一樣，那基本上就直接action dimension * 有幾個neighbor就是總dimension數量  # 要計算n_na_sparse 就是因為critic最後是會參照所有agent的
            if self.identical:
                n_na_sparse = self.n_a*n_n
            # 麻煩的總是沒有identical，有人一樣有人不一樣。記得嗎? na_dim_ls是一個list，他會存這個agent的鄰居的action維度，所以把它
            # sum起來就是鄰居action總維度，好棒呦
            else:
                n_na_sparse = sum(self.na_dim_ls)
            # 所以鄰居的action總維度也會當成hidden layer, 所以要加進hiddn layer的dimension當中
            n_h += n_na_sparse
        # 這個地方可以看到，他是一個線性層然後是從剛剛的hidden dimension到1
        # BUT....為何是1?
        # 定義 critic_head (class attribute)
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    # calculates the critic’s output given inputs (hidden state h and neighbor actions na
    # 輸入  神經網路的輸入資料(LSTM的輸出) 和鄰居以及自己的action
    # 輸出  預測累積獎勵的 scalar 值
    # 作用  把h送進critic_head 做運算並輸出預期獎勵
    # h 就是agent自己的LSTM output，na是鄰居預計採取的action
    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()  # na本來是numpy array，我們把他轉成tensor形式
            if self.identical:
                # 把鄰居的action動作進行one-hot編碼之後攤平
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a*n_n)
            else:
                # 如果不identical，鄰居的action dimension不同，就會超麻煩
                # 用文字不好表示，我將GPT的回答放https://drive.google.com/file/d/1BJF7NxvZvSun4ywGVqCFzNuRU8BXxXjj/view?usp=drive_link
                # 可自行搭配程式理解
                # Neighbor actions (na) are one-hot encoded and concatenated:
                # 𝑛𝑎_𝑠𝑝𝑎𝑟𝑠𝑒=[one-hot(Neighbor 1)]+[one-hot(Neighbor 2)]   ### Combine h (agent's hidden layer output) and na_sparse (neighbor actions).
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(
                        one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)

            h = torch.cat([h, na_sparse.cuda()], dim=1)
        # The critic_head(h) computes the final scalar output, which is the predicted cumulative reward.
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        # 這坨基本上就是做出碩論pp31pp32的演算法3.3&3.4
        # 主要有三種loss，actor、正則項、critic的loss

        # 可能需要的資訊有：
        # actor_dist：動作的概率分佈(policy π) π(a∣s)
        # e_coef：熵損失的權重係數
        # v_coef：價值損失的權重係數
        # vs：價值估計
        # As：實際採取的動作
        # Rs：目標價值（回報）
        # Advs：優勢函數  ⋅A(s,a)
        As = As.cuda()
        Advs = Advs.cuda()
        Rs = Rs.cuda()

        # 這兩行就是actor的loss(policy loss)
        # Lpolicy = −E[logπ(a∣s)⋅A(s,a)]  -----3.3 前面那一項

        log_probs = actor_dist.log_prob(As)  # logπ(a∣s)
        policy_loss = -(log_probs * Advs).mean()  # A(s,a)

        # Entropy Loss  正則項，增加策略的探索性，防止策略過早陷入局部最優  -----3.3後面那一項
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef

        # L value=E[(R−V(s))2] ----3.4
        value_loss = (Rs - vs).pow(2).mean() * v_coef

        # 這裡的 e_coef 和 v_coef 是用來調整熵損失和價值損失對總損失的影響程度。
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        # 這我先跳過
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(
            n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        #
        # n_fc：全連接層的單元數
            # n_lstm：LSTM 層的單元數
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        obs = obs.cuda()
        dones = dones.cuda()
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)

        # 下面是在計算loss，
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(
            logits=F.log_softmax(self.actor_head(hs), dim=1))  # hs丟入actor_head，再經過soft max
        vs = self._run_critic_head(hs, nactions)  # hs 以及自己與鄰居的動作丟入critic_head
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + \
            self.entropy_loss  # 總共的loss，經過run_loss之後得到三種loss，再把他們加起來
        self.loss.backward()  # 計算gradient，只差沒有optimizer.step() 更新NN參數
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().cuda()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().cuda()
        x = self._encode_ob(ob)  # fc
        h, new_states = run_rnn(
            self.lstm_layer, x, done, self.states_fw)  # lstm
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            # actor head
            return F.softmax(self.actor_head(h), dim=1).squeeze().cpu().detach().numpy()
        else:
            # critic head
            return self._run_critic_head(h, np.array([naction])).cpu().detach().numpy()

    def _encode_ob(self, ob):
        # fc_layer 是在下面init_net(self)中定義的，下面定義(初始化)了各種神經網路，比如說要多少neauron拉，輸入維度拉
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        # 這邊再統整一下模型中用到的神經網路
        # 1. 全連接層fc_layer: 用於把hidden_layer投射到特徵空間
        # 2. LSTM就是LSTM
        # 3. 策略頭，說到這個，講一下，「頭」是神經網路中的專業用語，這個網路有身體，然後他的輸出端有兩個，就是兩個「頭」
        #    策略頭的目的在於輸出動作的機率分布
        # 4. 評論頭，評論頭就是會輸出這個...value
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_lstm * 2)
        self.states_bw = torch.zeros(self.n_lstm * 2)





class NCMultiAgentPolicy(Policy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(
            n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

        # Initialize Separate Self-Attention Layers for Each Agent
        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_agent):
            attention_layer = nn.MultiheadAttention(
                embed_dim=self.n_h, num_heads=1)
            init_layer(attention_layer, 'fc')  # Ensure proper initialization
            self.attention_layers.append(attention_layer)

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(
                logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                               acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        # h dim: NxTxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

# 不用改
    def _get_comm_s(self, i, n_n, x, h, p):
        # 這method主要針對單一agent的LSTM的輸入做處理，具體內容如下
        # 1.收集鄰居的資訊（觀測值、隱藏狀態、策略指紋）
        # 2.對這些資訊進行處理
        # 3.通過全連接層（fc_x_layers、fc_p_layers、fc_m_layers）處理，並應用 ReLU 激活函數
        # 4.將處理後的特徵拼接，作為 LSTM 的輸入

        
        h = h.cuda()    # h是所有agent的hidden state(所有 LSTM layers t-1時刻的狀態) have a fixed size (n_h) for all agents
        x = x.cuda()    # x是所有agent目前的觀察值 (ob)
        p = p.cuda()    # p是所有agent的policy fingerprint(t-1時刻的所有actor_head輸出)
        ########################################################################################前面h,x,p都是所有agent的，現在要擷取鄰居的
        # js是目前agnet的鄰居的index，For example, if self.neighbor_mask[i] = [0, 1, 0, 1], it means agents 1 and 3 are neighbors of agent i.
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().cuda()   #js把目前agent的鄰居是有誰，用index存成一個list
        
        m_i = torch.index_select(h, 0, js)   ###根據剛剛得到的鄰居index(js)，現在從全部的hidden state(h)中要篩選出鄰居的hidden state，作為FC_m_layers的輸入(m_i)
        p_i = torch.index_select(p, 0, js)   # p_i 是鄰居在t-1時的policy fingerprint
        nx_i = torch.index_select(x, 0, js)  # nx_i 是鄰居在t時刻的觀測值

        m_i = m_i.view(1, self.n_h * n_n)    ###這裡直接把鄰居的hidden_state，展平成一維向量，形狀為 (1, self.n_h * n_n)。因為LSTM的output所有agent都一樣，不分identical or not

        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            p_i_ls = []  # 這個list用來存個鄰居的fingerprint
            nx_i_ls = []  # 這個則是鄰居的observation
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))),  # 第一項輸入，自己與鄰居的觀察。鄰居的觀測值（nx_i)。當前代理的觀測值（x_i)
               # 鄰居策略摘要（p_i）經 fc_p_layer 處理後，成為當前代理的輸入特徵之一 第三項輸入
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]  # 第二項輸入
        return torch.cat(s_i, dim=1)  # si 是LSTM的輸入

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        ''' 
        n_n: Number of neighbors.
        n_ns: Combined state dimension of the agent and its neighbors.
        n_na: Combined action dimension of the neighbors.
        '''
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        # Initialize existing layers
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []

        # Initialize Separate Attention Layers for Each Agent
        self.attention_layers = nn.ModuleList()

        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

            # Initialize the self-attention layer for each agent
            attention_layer = nn.MultiheadAttention(
                embed_dim=self.n_h, num_heads=1)
            init_layer(attention_layer, 'fc')
            self.attention_layers.append(attention_layer)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](
                    hs[i]), dim=1).squeeze().cpu().detach().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps
####################################################################################################  KEY
###這個function就是處理了所有agent的輸入，輸出是LSTM output，所以這個function的下一步就是進入actor head跟critic head 了。在forward中，就是先呼叫run_comm_layers，然後就進入_run_actor_heads跟_run_critic_head了！！
# fps 存了全部agent 在t-1時的policy(非action唷！)
# p 則是從fps中選出當前agent鄰居的policy

####################################################################################################
# _run_comm_layers Function
#
# Purpose:
# This function processes the inputs for all agents at each time step, updates their LSTM states 
# (hidden and cell states), and applies self-attention to the updated hidden states. The output 
# of this function is the LSTM hidden states after being processed by self-attention.
#
# Inputs:
# - obs   : Tensor of shape (T, n_agent, n_s) - Observations for all agents over T time steps.
# - dones : Tensor of shape (T, n_agent)      - Done flags indicating if an episode ended for each agent.
# - fps   : Tensor of shape (T, n_agent, n_a) - Policy fingerprints (previous policies) for all agents.
# - states: Tensor of shape (n_agent, 2 * n_h) - Combined hidden and cell states for all agents.
#
# Outputs:
# - outputs: Tensor of shape (T, n_agent, n_h) - The LSTM outputs (hidden states) for all agents over T time steps,
#            after applying self-attention.
# - new_states: Tensor of shape (n_agent, 2 * n_h) - The updated hidden and cell states for all agents.
#
# Next Steps:
# The output of this function (LSTM outputs) is passed into the actor head and critic head to generate:
# - The action probabilities (policy) via the actor head.
# - The value estimation via the critic head.
####################################################################################################
 
    def _run_comm_layers(self, obs, dones, fps, states):
        
        obs = batch_to_seq(obs) #agnet第一項輸入 ###all map's observation
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps) #agent第三項輸入 ### all agent's output policy

        #agent第二項輸入
        h, c = torch.chunk(states, 2, dim=1)  ###state是前一時刻LSTM的Hidden State (h)+Cell State (c)串接起來，變成一維向量。
        h = h.to(self.device)   #short-term memory or output
        c = c.to(self.device)   #long-term memory
        outputs = []

        for t, (x, p, done) in enumerate(zip(obs, fps, dones)): ##For each time step t, process the observations, policy fingerprints, and done flags.
            ##注意！在送入get_comm_s之前，所有的obs,fps,h,c都是所有agent，所有十字路口綜合起來一起存在一個list裡的！是把第i位agent送入get_comm_s後，第i位agent才去全部的x(obs),p(fps),(h,c)找他自己鄰居的來用！
            done = done.to(self.device)
            x = x.to(self.device)
            p = p.to(self.device)
            #Prepare Lists to Store Updated Hidden and Cell States(LSTM output)
            next_h = []  
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
        ## Process each agent separately
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    ####get_comm_s裡面有三個FC，其實合理，因為 _run_comm_layers是在forward 裡被呼叫的，所以在下面這部，就是第一次真正把輸入流過NN
                    s_i = self._get_comm_s(i, n_n, x, h, p)   ##return s_i。當前agent number(i)、n_n(number of neighbors)、x(all agent's observation),h(all LSTM's hidden state),p(all agents policy)丟進去get_comm_s，他就會自動篩選出鄰居的資訊。得到符合此位agent(agent i)的專屬s_i
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                ####執行到這裡，就準備把s_i餵入LSTM了！
                '''
                LSTM Forward Pass
At each time step, the LSTM takes:

1.Input (x_t): The current input (feature vector).
2.Previous Hidden State (h_t-1): The hidden state from the previous time step.
3.Previous Cell State (c_t-1): The cell state from the previous time step.

The LSTM updates these states and produces:

1.New Hidden State (h_t): The output of the LSTM at the current time step.(除了是short term output，也是預測的目標output)
2.New Cell State (c_t): The updated long-term memory.
                '''
                ##整理一下要餵入LSTM的東西
                ##1. s_i：經過FC處理後的所有agent能觀察到的訊息
                ##2. h[i] : current hidden  states for agent i
                ##3. c[i] : current cell    states for agent i
                ##unsqueeze(0)： adds an extra dimension to make their shape (1, n_h) (suitable for LSTM input).
                ####LSTM 層根據當前的輸入 s_i 和先前的隱藏狀態(h_i)以及細胞狀態(c_i)。來計算新的隱藏狀態 next_h_i 和細胞狀態 next_c_i 
                h_i, c_i = h[i].unsqueeze(0) * (1 - done), c[i].unsqueeze(0) * (1 - done)
                ###輸入流過的第二個NN(LSTM)
                ###LSTM的輸出請看 ： https://chatgpt.com/share/675b22d9-4400-8013-8102-1a28d8c13ffe
                '''
                Key Context to Differentiate
                lstm_out, (h_n, c_n) = self.lstm(x)   # lstm_out contains hidden states for all time steps: shape (batch_size, seq_length, hidden_size)

                接下來分為兩種使用LSTM output的方式
                Sequence-to-One: If after running this LSTM operation, you only take the last hidden state (next_h_i[-1]) to pass into a fully connected layer for a single prediction.(e,g最後一天股價)
                if we want to get single value prediction for the entire sequence:
                final_hidden_state = lstm_out[:, -1, :] 

                
                Sequence-to-Sequence: If you collect all hidden states (next_h_i at each time step) and produce outputs for each step.(e.g.文章每個字詞性分析)
                # in init ： self.fc = nn.Linear(hidden_size, output_size)
                out = self.fc(lstm_out)          # Fully connected layer applied to each time step
                return out
                https://chatgpt.com/share/675b22d9-4400-8013-8102-1a28d8c13ffe
                '''
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))  #next_h_i：更新後的隱藏狀態(新的short term memory！，同時也是LSTM的final output！)，將在下一步作為 Self-Attention 的輸入。
                next_h.append(next_h_i)#回想LSTM的圖片，這就等於是長長的short term memory往右邊又多了一筆資料。往右邊多新增了一個LSTM block
                next_c.append(next_c_i)#長長的long term memory往右邊又多了一筆資料
            # after LSTM cell updates
            h = torch.cat(next_h) # shape: (n_agent, n_h)
            c = torch.cat(next_c)

            # Apply each agent's self-attention layer to its hidden state
            # Apply self-attention layer for each agent
            h_seq = h.unsqueeze(1)  # (n_agent, n_h) --> (n_agent, 1, n_h)
            attn_output_list = []
            for i in range(self.n_agent):
                attn_output, _ = self.attention_layers[i](h_seq[i].unsqueeze(0), h_seq[i].unsqueeze(0), h_seq[i].unsqueeze(0))
                attn_output_list.append(attn_output.squeeze(0))
            h = torch.cat(attn_output_list, dim=0)
            outputs.append(h.unsqueeze(0))


            # Concatenate all agents' attention outputs
            h = torch.stack(attn_output_list)  # Shape: (n_agent, n_h)
            outputs.append(h.unsqueeze(0))    # Shape: (1, n_agent, n_h)

    
        outputs = torch.cat(outputs)           # Shape: (batch, n_agent, n_h)
        return outputs, torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(
                    np.where(self.neighbor_mask[i])[0]).long()
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(
                        one_hot(na_i[j], self.na_ls_ls[i][j]).cuda())
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.cpu().detach().numpy())
            else:
                vs.append(v_i)
        return vs

#####


class NCLMMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, groups=0, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'nclm', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.groups = groups
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        bps = self._run_actor_heads(hs, acts)
        # for i in range(self.n_agent):
        #     if i % 2 != 0:
        #         ps[i] = bps[i]
        for i in range(self.n_agent):
            if i in self.groups:
                ps[i] = bps[i]

        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(
                logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                               acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        # TxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()
        # h dim: NxTxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            if (np.array(action) != None).all():
                action = torch.from_numpy(
                    np.expand_dims(action, axis=1)).long()
            return self._run_actor_heads(h, action, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        # fc_x_layer 本地觀測值得全連接層
        fc_x_layer = nn.Linear(n_ns, self.n_fc)

        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            # fc_p_layer 鄰居的策略指紋 (就是t-1時鄰居的policy)
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            # fc_m_layer 鄰居的訊息指紋
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_backhand_actor_head(self, n_a, n_na):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h + n_na, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            if i not in self.groups:
                # first move
                self._init_actor_head(n_a)
            else:
                self._init_backhand_actor_head(n_a, n_na)
            self._init_critic_head(n_na)

    def _run_actor_heads(self, hs, preactions=None, detach=False):
        ps = [0] * self.n_agent
        if (np.array(preactions) == None).all():
            for i in range(self.n_agent):
                if i not in self.groups:  # first hand
                    if detach:
                        p_i = F.softmax(self.actor_heads[i](
                            hs[i]), dim=1).cpu().squeeze().detach().numpy()
                    else:
                        p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
                    ps[i] = p_i
        else:
            for i in range(self.n_agent):
                if i in self.groups:  # back hand
                    n_n = self.n_n_ls[i]
                    if n_n:
                        js = torch.from_numpy(
                            np.where(self.neighbor_mask[i])[0]).long()
                        na_i = torch.index_select(preactions, 0, js)
                        na_i_ls = []
                        for j in range(n_n):
                            na_i_ls.append(
                                one_hot(na_i[j], self.na_ls_ls[i][j]).cuda())
                        h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
                    else:
                        h_i = hs[i]
                    if detach:
                        p_i = F.softmax(self.actor_heads[i](
                            h_i), dim=1).cpu().squeeze().detach().numpy()
                    else:
                        p_i = F.log_softmax(self.actor_heads[i](h_i), dim=1)
                    ps[i] = p_i
        return ps
