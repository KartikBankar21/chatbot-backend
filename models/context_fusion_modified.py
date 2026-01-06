# context_fusion.py (modified)
import torch
import torch.nn as nn

class ContextFusion(nn.Module):
    def __init__(self, hidden_dim, context_window=3, num_heads=4, dropout=0.3):
        super(ContextFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        
        # VALIDATION: Check that hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            # Auto-adjust num_heads to nearest valid value
            valid_heads = [h for h in [1, 2, 3, 4, 6, 8, 12, 16] if hidden_dim % h == 0]
            if valid_heads:
                num_heads = max([h for h in valid_heads if h <= num_heads])
                print(f"⚠️  Auto-adjusted num_heads to {num_heads} (hidden_dim={hidden_dim})")
            else:
                raise ValueError(
                    f"Cannot find valid num_heads for hidden_dim={hidden_dim}. "
                    f"hidden_dim must be divisible by num_heads."
                )
        
        self.turn_pos_emb = nn.Embedding(context_window, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
    def forward(self, turn_vecs):
        batch, K, D = turn_vecs.size()
        pos = torch.arange(K, device=turn_vecs.device)
        pos_emb = self.turn_pos_emb(pos)
        pos_emb = pos_emb.unsqueeze(0).expand(batch, K, D)
        turn_vecs = turn_vecs + pos_emb
        attn_output, _ = self.attn(turn_vecs, turn_vecs, turn_vecs)
        context_vec = attn_output.mean(dim=1)
        return context_vec

