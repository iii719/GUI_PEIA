import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, user, mood, Genre):
        self.input_data = {'user': user, 'mood': mood, 'Genre': Genre}

    def __getitem__(self, index):
        input_data = {k: {sub_k: v[sub_k][index] for sub_k in v.keys()} for k, v in self.input_data.items()}
        return input_data

    def __len__(self):
        return len(next(iter(self.input_data['user'].values())))

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(num_hidden_layers)],
            nn.Linear(64, hidden_size),
        )
    def forward(self, x):
        return self.layers(x)

class Score(nn.Module):
    def __init__(self, d_k):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_k, d_k),
            nn.Sigmoid(),
            nn.Linear(d_k, 1),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear = nn.Linear(d_model, d_model)
        self.l_to_score, self.s_to_score, self.zl_to_score, self.zs_to_score = [Score(self.d_k) for _ in range(4)]

    def split_heads(self, x,):
        return x.view(-1, self.num_heads, self.d_k)

    def group_heads(self, x,):
        return x.contiguous().view(-1, self.num_heads * self.d_k)

    def summing_scores(self, scores, original):
        stacked_scores = torch.stack([scores[k] * original[k] for k in scores.keys()])
        return torch.sum(stacked_scores, dim=0)

    def forward(self, user_embedded, mood_embedded, Genre_embedded):

        L = {u_key + '_' + s_key: self.split_heads(u_value * s_value)/np.sqrt(self.d_k) for u_key, u_value in user_embedded.items() for s_key, s_value in Genre_embedded.items()}
        L_score = {key: self.l_to_score(value) for key,value in L.items()}

        S = {m_key + '_' + s_key: self.split_heads(m_value * s_value)/np.sqrt(self.d_k) for m_key, m_value in mood_embedded.items() for s_key, s_value in Genre_embedded.items()}
        S_score = {key: self.s_to_score(value) for key,value in S.items()}

        L_att = self.summing_scores(L_score, L)
        S_att = self.summing_scores(S_score, S)

        rl = self.zl_to_score(L_att)
        rs = self.zs_to_score(S_att)
        r = torch.cat([rl, rs], dim=-1)

        r = f.softmax(r, dim=-1)
        rl, rs = r.chunk(2, dim=-1)

        out = rl * L_att + rs * S_att

        out = self.group_heads(out)

        out = self.linear(out)

        return out

class Embeddings(nn.Module):
    def __init__(self, d_model, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input):
        linear_output = self.linear(input)
        embeddings = self.LayerNorm(linear_output)
        return embeddings

class Embedder(nn.Module):
    def __init__(self, d_model, user_keys, mood_keys, Genre_keys):
        super().__init__()

        self.user_embeddings = nn.ModuleDict({k: Embeddings(d_model, 1) for k in user_keys})
        self.mood_embeddings = nn.ModuleDict({k: Embeddings(d_model, 1) for k in mood_keys})
        self.Genre_embeddings = nn.ModuleDict({k: Embeddings(d_model, 1) for k in Genre_keys})

    def forward(self, user, mood, Genre):
        user_embedded = {k: emb(user[k]) for k, emb in self.user_embeddings.items()}
        mood_embedded = {k: emb(mood[k]) for k, emb in self.mood_embeddings.items()}
        Genre_embedded = {k: emb(Genre[k]) for k, emb in self.Genre_embeddings.items()}

        return user_embedded, mood_embedded, Genre_embedded

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, num_heads, user_keys, mood_keys, Genre_keys, num_hidden_layers):
        super().__init__()

        self.mother_tongue_pre_embed = nn.Embedding(num_embeddings=40, embedding_dim=1)
        self.all_keys_number = len(user_keys)+len(mood_keys)+len(Genre_keys)

        self.embedder = Embedder(d_model,user_keys, mood_keys, Genre_keys)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dnn = DNN(self.all_keys_number*d_model, d_model, num_hidden_layers)

        self.W_zd = nn.Linear(d_model, 10, bias=False)
        self.W_z0 = nn.Linear(self.all_keys_number, 10, bias=False)
        self.W_zatt = nn.Linear(d_model, 10, bias=False)

        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, user, mood, Genre):
        for k in user.keys():
          user[k] = user[k].unsqueeze(1)
        for k in mood.keys():
          mood[k] = mood[k].unsqueeze(1)
        for k in Genre.keys():
          Genre[k] = Genre[k].unsqueeze(1)

        user['mother tongue'] = self.mother_tongue_pre_embed(user['mother tongue']).squeeze(1)

        user_embedded, mood_embedded, Genre_embedded = self.embedder(user, mood, Genre)

        all_embedded_values = torch.cat([*user_embedded.values(), *mood_embedded.values(), *Genre_embedded.values()], dim=1)

        zd = self.dnn(all_embedded_values)

        z0 = torch.cat((*user.values(),  *mood.values(), *Genre.values()), dim=1)

        zatt = self.mha(user_embedded,mood_embedded, Genre_embedded)

        out = torch.sigmoid(self.W_zd(zd) + self.W_z0(z0) + self.W_zatt(zatt) + self.bias)

        return out
    
user_keys = ['age', 'gender', 'mother tongue']
mood_keys = ['mood']
Genre_keys = ['genre1', 'genre2']

def load_model():
    model = TransformerClassifier(d_model=32, num_heads=2, user_keys=user_keys, mood_keys=mood_keys, Genre_keys = Genre_keys, num_hidden_layers=2)
    model.load_state_dict(torch.load('static/model_parameters_label.pt', map_location=torch.device('cpu')))
    model.eval()
    return model