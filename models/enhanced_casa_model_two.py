"""
Enhanced CASA Model with Requested Slot Prediction
Adds a new head for predicting which slots the user is asking about
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.disan import DiSAN
from models.context_fusion_modified import ContextFusion


class CASA_NLU_Standalone(nn.Module):
    """
    Standalone version with all dependencies inline and dimension validation
    """
    
    def __init__(self, vocab_size, intent_size, slot_size, da_size,
                 num_requestable_slots=20,
                 hidden_dim=128, embed_dim=128,
                 intent_embed_dim=16, da_embed_dim=16,  # FIXED: Changed from 20, 10
                 context_window=3, sliding_window=3, 
                 dropout=0.3, num_heads=4):
        super(CASA_NLU_Standalone, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.sliding_window = sliding_window
        
        # 1. Encoders
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.disan = DiSAN(embed_dim, hidden_dim, dropout=dropout)
        self.disan_output_dim = 2 * hidden_dim
        
        # 2. History Embeddings
        self.intent_hist_emb = nn.Embedding(intent_size, intent_embed_dim)
        self.da_hist_emb = nn.Embedding(da_size, da_embed_dim)
        
        # 3. Context Fusion - CRITICAL: Calculate and validate dimensions
        fusion_input_dim = self.disan_output_dim + intent_embed_dim + da_embed_dim
        
        print(f"🔧 Context Fusion Dimensions:")
        print(f"   DiSAN output: {self.disan_output_dim}")
        print(f"   Intent embed: {intent_embed_dim}")
        print(f"   DA embed: {da_embed_dim}")
        print(f"   Fusion input: {fusion_input_dim}")
        print(f"   Num heads: {num_heads}")
        print(f"   Divisibility check: {fusion_input_dim} % {num_heads} = {fusion_input_dim % num_heads}")
        
        if fusion_input_dim % num_heads != 0:
            raise ValueError(
                f"Fusion input dim ({fusion_input_dim}) must be divisible by num_heads ({num_heads}).\n"
                f"Current: disan({self.disan_output_dim}) + intent({intent_embed_dim}) + da({da_embed_dim}) = {fusion_input_dim}\n"
                f"Try: intent_embed_dim=16, da_embed_dim=16 for num_heads=4"
            )
        
        self.context_fusion = ContextFusion(
            fusion_input_dim, context_window, num_heads=num_heads, dropout=dropout
        )
        
        # 4. IC Layers
        self.secondary_intent_fc = nn.Linear(self.disan_output_dim, intent_size)
        self.intent_fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.intent_fc2 = nn.Linear(hidden_dim + self.disan_output_dim, intent_size)
        
        # 5. Slot Layers
        self.slot_emb = nn.Embedding(slot_size, hidden_dim)
        self.fusion_gate = nn.Linear(2 * self.disan_output_dim, self.disan_output_dim)
        
        ic_penult_dim = hidden_dim
        slot_hist_dim = hidden_dim
        slot_input_dim = (self.disan_output_dim * sliding_window) + slot_hist_dim + ic_penult_dim
        
        self.slot_gru = nn.GRU(slot_input_dim, hidden_dim, batch_first=True)
        self.slot_classifier = nn.Linear(hidden_dim, slot_size)
        
        # 6. NEW: Requested Slot Prediction
        req_slot_input_dim = self.disan_output_dim + fusion_input_dim
        self.requested_slot_fc1 = nn.Linear(req_slot_input_dim, hidden_dim)
        self.requested_slot_classifier = nn.Linear(hidden_dim, num_requestable_slots)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, utterance, turn_utterance_history, intent_history, da_history, slot_history=None):
        # 1. Current Utterance Encoding
        x = self.embedding(utterance)
        utt_vec, token_reps = self.disan(x)
        
        # 2. Secondary IC
        secondary_intent_logits = self.secondary_intent_fc(utt_vec)
        
        # 3. Signal Encoding & Context Fusion
        h_intent = self.intent_hist_emb(intent_history)
        h_da = self.da_hist_emb(da_history)
        T_i = torch.cat([turn_utterance_history, h_intent, h_da], dim=2)
        context_vec = self.context_fusion(T_i)
        
        # 4. Primary Intent Classification
        h_int = torch.tanh(self.intent_fc1(context_vec))
        intent_input = torch.cat([h_int, utt_vec], dim=1)
        intent_logits = self.intent_fc2(intent_input)
        
        # 5. Slot Labeling
        seq_len = token_reps.size(1)
        utt_expanded = utt_vec.unsqueeze(1).expand(-1, seq_len, -1)
        fusion_cat = torch.cat([token_reps, utt_expanded], dim=2)
        gate = torch.sigmoid(self.fusion_gate(fusion_cat))
        fused_tokens = gate * token_reps + (1 - gate) * utt_expanded
        
        pad = (self.sliding_window - 1) // 2
        padded = F.pad(fused_tokens, (0, 0, pad, pad))
        sliding_feats = []
        for i in range(seq_len):
            win = padded[:, i:i+self.sliding_window, :]
            sliding_feats.append(win.reshape(win.size(0), -1))
        sliding_feats = torch.stack(sliding_feats, dim=1)
        
        if slot_history is not None:
            slot_hist_embed = self.slot_emb(slot_history).mean(dim=1)
        else:
            slot_hist_embed = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            
        ic_feat = h_int
        slot_input = torch.cat([
            sliding_feats,
            slot_hist_embed.unsqueeze(1).expand(-1, seq_len, -1),
            ic_feat.unsqueeze(1).expand(-1, seq_len, -1)
        ], dim=2)
        
        gru_out, _ = self.slot_gru(slot_input)
        slot_logits = self.slot_classifier(gru_out)
        
        # 6. Requested Slot Prediction
        req_input = torch.cat([utt_vec, context_vec], dim=1)
        req_hidden = torch.tanh(self.requested_slot_fc1(req_input))
        requested_slot_logits = self.requested_slot_classifier(req_hidden)
        
        return intent_logits, slot_logits, secondary_intent_logits, requested_slot_logits


# ============================================================================
# WRAPPER
# ============================================================================
class CASATrainerWrapperStandalone(nn.Module):
    def __init__(self, casa_model):
        super().__init__()
        self.model = casa_model

    def forward(self, tokens, hist_tokens, hist_intents, hist_das):
        B, W, L = hist_tokens.size()
        flat_hist = hist_tokens.view(B * W, L)
        hist_emb = self.model.embedding(flat_hist)
        hist_vecs, _ = self.model.disan(hist_emb)
        hist_vecs_reshaped = hist_vecs.view(B, W, -1)
        
        intent_logits, slot_logits, sec_logits, req_slot_logits = self.model(
            utterance=tokens,
            turn_utterance_history=hist_vecs_reshaped,
            intent_history=hist_intents,
            da_history=hist_das,
            slot_history=None
        )
        
        return intent_logits, slot_logits, sec_logits, req_slot_logits


# ============================================================================
# INITIALIZATION FUNCTION
# ============================================================================
def create_requested_slot_mapping(all_schemas):
    """Create mapping for requestable slots"""
    requestable_slots = set()
    
    if isinstance(all_schemas, list):
        schemas = {s['service_name']: s for s in all_schemas}
    else:
        schemas = all_schemas
    
    for service_name, schema in schemas.items():
        slots = schema.get('slots', {})
        
        if isinstance(slots, dict):
            slot_names = slots.get('name', [])
        else:
            slot_names = [s['name'] for s in slots]
        
        requestable_slots.update(slot_names)
    
    requestable_slots = sorted(list(requestable_slots))
    slot2id = {slot: i for i, slot in enumerate(requestable_slots)}
    id2slot = {i: slot for slot, i in slot2id.items()}
    
    return slot2id, id2slot, len(requestable_slots)

def extract_requested_slot_labels(raw_frame, req_slot2id):
    """
    Extract multi-hot vector for requested slots from ground truth.
    
    Args:
        raw_frame: Ground truth frame
        req_slot2id: Mapping from slot name to ID
        
    Returns:
        Tensor of shape (num_requestable_slots,) with 1s for requested slots
    """
    num_slots = len(req_slot2id)
    labels = torch.zeros(num_slots)
    
    if raw_frame.get('state') and len(raw_frame['state']) > 0:
        state = raw_frame['state'][0]
        requested_slots = state.get('requested_slots', [])
        
        for slot_name in requested_slots:
            if slot_name in req_slot2id:
                labels[req_slot2id[slot_name]] = 1.0
    
    return labels

def initialize_standalone_enhanced_model(
    vocab_size, intent_size, slot_size, da_size,
    all_schemas, device='cuda'
):
    """
    Initialize standalone enhanced model with full validation.
    """
    print("="*80)
    print("🚀 Initializing Standalone Enhanced CASA")
    print("="*80)
    
    # Create requested slot mapping
    req_slot2id, req_id2slot, num_req_slots = create_requested_slot_mapping(all_schemas)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Vocab: {vocab_size}")
    print(f"   Intents: {intent_size}")
    print(f"   Slot Tags (BIO): {slot_size}")
    print(f"   Dialog Acts: {da_size}")
    print(f"   Requestable Slots: {num_req_slots}")
    
    # Initialize model
    casa_core = CASA_NLU_Standalone(
        vocab_size=vocab_size,
        intent_size=intent_size,
        slot_size=slot_size,
        da_size=da_size,
        num_requestable_slots=num_req_slots,
        hidden_dim=128,
        embed_dim=128,
        intent_embed_dim=16,  # FIXED: 16 instead of 20
        da_embed_dim=16,      # FIXED: 16 instead of 10
        context_window=3,
        sliding_window=3,
        dropout=0.3,
        num_heads=4
    )
    
    # Wrap
    model = CASATrainerWrapperStandalone(casa_core)
    model = model.to(device)
    
    req_mappings = {
        'req_slot2id': req_slot2id,
        'req_id2slot': req_id2slot,
        'num_req_slots': num_req_slots
    }
    
    print(f"\n✅ Standalone model initialized on {device}")
    print("="*80)
    
    return model, req_mappings
