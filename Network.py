import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim

from time import time
from datetime import datetime


class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long()


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class Act_net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(params["num_of_stages"], params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
        self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.W_q2 = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref2 = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.dec_input = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.use_logit_clipping = params["use_logit_clipping"]
        self.C = params["C"]
        self.T = params["T"]
        self.n_glimpse = params["n_glimpse"]
        self.block_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device, y=None):
        x = x.to(device)
        batch, block_num, stage_number = x.size()
        # x: batch, job number, stage number(processing time)

        embed_enc_inputs = self.Embedding(x)
        # embed_enc_inputs: batch, job number, n_embedding)

        embed_num = embed_enc_inputs.size(2)
        # embed_num: n_embedding

        mask = torch.zeros((batch, block_num), device=device)

        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        # h (1, batch, hidden_size)
        # c (1, batch, hidden_size)
        # enc_h (batch,job_number, hidden_size)
        ref = enc_h
        pi_list, log_ps = [], []

        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(device)
        # dec_input (batch,1,hidden_size)

        #
        for i in range(block_num):
            dec_h, (h, c) = self.Decoder(dec_input, (h, c))
            query = h.squeeze(0)
            for j in range(self.n_glimpse):
                query = self.glimpse(query, ref, mask)
            logits = self.pointer(query, ref, mask)

            # 50, 40 (batch, job number)
            log_p = torch.log_softmax(logits / self.T, dim=-1)
            # query.detach().cpu()
            # logits.detach().cpu()
            if y == None:
                next_block = self.block_selecter(log_p)
            else:
                next_block = y[:, i].long()
            dec_input = torch.gather(input=embed_enc_inputs, dim=1,
                                     index=next_block.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed_num))
            pi_list.append(next_block)
            log_ps.append(log_p)
            mask += torch.zeros((batch, block_num), device=device).scatter_(dim=1, index=next_block.unsqueeze(1),
                                                                            value=1)

        seq = torch.stack(pi_list, dim=1)
        ps = torch.stack(log_ps, dim=1)
        ll = self.get_log_likelihood(torch.stack(log_ps, 1), seq)

        return seq, ll, ps
        # seq (50,40)
        # ll (50)
        # ps (50 40 40) (batch, 순서, 그떄 확률) (0,0,:)=1

    def glimpse(self, query, ref, mask, inf=1e8):
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, hidden_size, block_num)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, hidden_size, block_num)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        # u1.detach().cpu()
        # u2.detach().cpu()
        # V.detach().cpu()

        u = u - inf * mask
        a = F.softmax(u, dim=1)
        g = torch.bmm(a.unsqueeze(1), ref).squeeze(1)
        # u2: (batch, 128, block_num) * a: (batch, block_num, 1) => d: (batch, hidden size)
        return g

    def pointer(self, query, ref, mask, inf=1e8):
        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, block_num)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # u1.detach().cpu()
        # u2.detach().cpu()
        # V.detach().cpu()

        if self.use_logit_clipping:
            u = self.C * torch.tanh(u)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        u = u - inf * mask
        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)


class Cri_net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Embedding = nn.Linear(params["num_of_stages"], params["n_embedding"], bias=False)
        self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
        self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
        self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
        self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
        self.final2FC = nn.Sequential(
            nn.Linear(params["n_hidden"], params["n_hidden"], bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(params["n_hidden"], 1, bias=False))
        self._initialize_weights(params["init_min"], params["init_max"])
        self.n_glimpse = params["n_glimpse"]
        self.n_process = params["n_process"]

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device):
        '''	x: (batch, block_num, process_num)
            enc_h: (batch, block_num, embed)
            query(Decoder input): (batch, 1, embed)
            h: (1, batch, embed)
            return: pred_l: (batch)
        '''
        x = x.to(device)
        embed_enc_inputs = self.Embedding(x)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        # enc_h (batch, job_number, hidden_size)
        query = h[-1]

        # h (1, batch, hidden_size)
        for i in range(self.n_process):
            query = self.glimpse(query, ref)

        pred_l = self.final2FC(query).squeeze(-1)
        return pred_l

    def glimpse(self, query, ref, infinity=1e8):
        """	Args:
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, city_t, 128)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128,block_num) => u: (batch, 1, block_num) => (batch, block_num)
        a = F.softmax(u, dim=1)
        g = torch.bmm(a.unsqueeze(1), ref).squeeze(1)
        # d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, block_num) * a: (batch, block_num, 1) => d: (batch, 128)
        return g