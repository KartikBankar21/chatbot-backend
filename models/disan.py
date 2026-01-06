import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# INLINE: DiSAN (from your existing code)
# ============================================================================
class DiSAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(DiSAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_emb = nn.Embedding(500, input_dim)
        
        self.W_q_fw = nn.Linear(input_dim, hidden_dim)
        self.W_k_fw = nn.Linear(input_dim, hidden_dim)
        self.W_v_fw = nn.Linear(input_dim, hidden_dim)
        
        self.W_q_bw = nn.Linear(input_dim, hidden_dim)
        self.W_k_bw = nn.Linear(input_dim, hidden_dim)
        self.W_v_bw = nn.Linear(input_dim, hidden_dim)
        
        self.gate_fw = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_bw = nn.Linear(2 * hidden_dim, hidden_dim)
        
        s2t_input_dim = 2 * hidden_dim
        self.s2t_W2 = nn.Linear(s2t_input_dim, s2t_input_dim)
        self.s2t_W1 = nn.Linear(s2t_input_dim, s2t_input_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, _ = x.size()
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        H_i = x + self.pos_emb(positions)
        
        def compute_attention(q, k, v, mask):
            scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            scores = scores.masked_fill(mask == 0, -1e9)
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            return torch.bmm(probs, v)

        mask_fw = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        H_mF = compute_attention(
            self.W_q_fw(H_i), self.W_k_fw(H_i), self.W_v_fw(H_i), mask_fw
        )

        mask_bw = torch.triu(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        H_mB = compute_attention(
            self.W_q_bw(H_i), self.W_k_bw(H_i), self.W_v_bw(H_i), mask_bw
        )
        
        cat_fw = torch.cat([H_i, H_mF], dim=2)
        G_fw = torch.sigmoid(self.gate_fw(cat_fw))
        C_F = G_fw * H_i + (1 - G_fw) * H_mF
        
        cat_bw = torch.cat([H_i, H_mB], dim=2)
        G_bw = torch.sigmoid(self.gate_bw(cat_bw))
        C_B = G_bw * H_i + (1 - G_bw) * H_mB
        
        C_i = torch.cat([C_F, C_B], dim=2)
        
        activ = torch.sigmoid(self.s2t_W2(C_i))
        scores = self.s2t_W1(activ)
        attn_weights = F.softmax(scores, dim=1)
        utt_vec = torch.sum(attn_weights * C_i, dim=1)
        
        return utt_vec, C_i
