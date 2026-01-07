"""
Model loading and inference management - ROBUST VERSION
"""
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import BertTokenizer

from models.enhanced_casa_model_two import (
    CASA_NLU_Standalone,
    CASATrainerWrapperStandalone,
    create_requested_slot_mapping
)
from models.dialogue_state_tracker_two import (
    enhanced_slot_value_extraction,
    extract_predicted_requested_slots
)
from app.config import config


class ModelManager:
    """
    Manages model loading, inference, and preprocessing
    """
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.model = None
        self.tokenizer = None
        self.mappings = None
        self.req_mappings = None
        self.loaded = False
    
    def load_model(self, checkpoint_path: Path, mappings_path: Path):
        """Load model and mappings with robust checkpoint format detection"""
        print(f"🔄 Loading model from {checkpoint_path}...")
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load mappings
        with open(mappings_path, 'rb') as f:
            self.mappings = pickle.load(f)
        
        print(f"📊 Loaded mappings:")
        print(f"   - Vocab size: {len(self.mappings['vocab'])}")
        print(f"   - Intents: {len(self.mappings['intent2id'])}")
        print(f"   - Slot tags: {len(self.mappings['slot2id'])}")
        print(f"   - Dialog acts: {len(self.mappings['da2id'])}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # ============================================================
        # DETECT CHECKPOINT FORMAT AND EXTRACT STATE DICT
        # ============================================================
        print(f"🔍 Detecting checkpoint format...")
        
        state_dict = None
        req_slot2id = None
        req_id2slot = None
        num_requestable_slots = None
        
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint is a dictionary with keys: {list(checkpoint.keys())[:5]}...")
            
            # Format 1: Dictionary with 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                req_slot2id = checkpoint.get('req_slot2id')
                req_id2slot = checkpoint.get('req_id2slot')
                num_requestable_slots = checkpoint.get('num_requestable_slots')
                print("✅ Format: Dictionary with 'model_state_dict' key")
            
            # Format 2: Dictionary with 'model' key
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                req_slot2id = checkpoint.get('req_slot2id')
                req_id2slot = checkpoint.get('req_id2slot')
                num_requestable_slots = checkpoint.get('num_requestable_slots')
                print("✅ Format: Dictionary with 'model' key")
            
            # Format 3: Direct state_dict (YOUR FORMAT)
            # Check if keys look like model parameters (e.g., 'model.embedding.weight')
            elif any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
                print("✅ Format: Direct state_dict (saved with model.state_dict())")
            
            else:
                raise ValueError(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")
        
        else:
            # Format 4: Entire model object (rare)
            raise ValueError(f"Checkpoint is not a dictionary, type: {type(checkpoint)}")
        
        if state_dict is None:
            raise ValueError("Could not extract state_dict from checkpoint")
        
        print(f"   State dict has {len(state_dict)} parameters")
        
        # ============================================================
        # CREATE MODEL ARCHITECTURE
        # ============================================================
        from app.schemas import DOMAIN_SCHEMAS
        
        # Determine num_requestable_slots
        if num_requestable_slots is None:
            if req_slot2id:
                num_requestable_slots = len(req_slot2id)
            else:
                # Count from schemas
                all_slots = set()
                for domain_schema in DOMAIN_SCHEMAS.values():
                    all_slots.update(domain_schema['slots'].keys())
                num_requestable_slots = len(all_slots)
                print(f"⚠️  No num_requestable_slots found, inferred {num_requestable_slots} from schemas")
        
        print(f"📊 Model configuration:")
        print(f"   - Vocab: {len(self.mappings['vocab'])}")
        print(f"   - Intents: {len(self.mappings['intent2id'])}")
        print(f"   - Slots: {len(self.mappings['slot2id'])}")
        print(f"   - Dialog Acts: {len(self.mappings['da2id'])}")
        print(f"   - Requestable Slots: {num_requestable_slots}")
        
        casa_core = CASA_NLU_Standalone(
            vocab_size=len(self.mappings['vocab']),
            intent_size=len(self.mappings['intent2id']),
            slot_size=len(self.mappings['slot2id']),
            da_size=len(self.mappings['da2id']),
            num_requestable_slots=num_requestable_slots,
            hidden_dim=config.HIDDEN_DIM,
            embed_dim=config.EMBED_DIM,
            intent_embed_dim=16,
            da_embed_dim=16,
            context_window=config.CONTEXT_WINDOW,
            sliding_window=config.SLIDING_WINDOW,
            dropout=config.DROPOUT,
            num_heads=config.NUM_HEADS
        )
        
        self.model = CASATrainerWrapperStandalone(casa_core)
        
        # ============================================================
        # LOAD STATE DICT WITH FALLBACK
        # ============================================================
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("✅ Loaded state dict (strict mode)")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed: {str(e)[:200]}")
            print("🔄 Trying non-strict loading...")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️  Missing keys: {missing[:5]}")
            if unexpected:
                print(f"⚠️  Unexpected keys: {unexpected[:5]}")
            print("✅ Loaded state dict (non-strict mode)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # ============================================================
        # SETUP REQUESTED SLOT MAPPINGS
        # ============================================================
        if req_slot2id and req_id2slot:
            self.req_mappings = {
                'req_slot2id': req_slot2id,
                'req_id2slot': req_id2slot,
                'num_req_slots': num_requestable_slots
            }
            print("✅ Loaded req_slot mappings from checkpoint")
        else:
            # Fallback: create from schemas
            print("⚠️  Creating req_slot mappings from schemas...")
            req_slot2id, req_id2slot, num_req_slots = create_requested_slot_mapping(DOMAIN_SCHEMAS)
            self.req_mappings = {
                'req_slot2id': req_slot2id,
                'req_id2slot': req_id2slot,
                'num_req_slots': num_req_slots
            }
            print(f"✅ Created {num_req_slots} req_slot mappings")
        
        self.loaded = True
        print(f"✅ Model loaded successfully on {self.device}")
        print("="*80)
    
    def tokenize_and_encode(self, utterance: str) -> Tuple[List[str], List[int]]:
        """Tokenize utterance and convert to IDs"""
        tokens = utterance.lower().split()
        
        # Convert to IDs using vocab
        token_ids = []
        for token in tokens:
            token_id = self.mappings['vocab'].get(token, self.mappings['vocab'].get('<UNK>', 0))
            token_ids.append(token_id)
        
        # Truncate if needed
        if len(token_ids) > config.MAX_TOKENS:
            tokens = tokens[:config.MAX_TOKENS]
            token_ids = token_ids[:config.MAX_TOKENS]
        
        return tokens, token_ids
    
    def prepare_history(self, history_utterances: List[str], 
                       history_intents: List[str],
                       history_dialog_acts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare history tensors for model input"""
        context_window = config.CONTEXT_WINDOW
        
        # Pad history to context_window
        while len(history_utterances) < context_window:
            history_utterances.insert(0, "")
            history_intents.insert(0, "NONE")
            history_dialog_acts.insert(0, "INFORM")
        
        # Take only recent context
        history_utterances = history_utterances[-context_window:]
        history_intents = history_intents[-context_window:]
        history_dialog_acts = history_dialog_acts[-context_window:]
        
        # Encode history utterances
        hist_tokens = []
        for utt in history_utterances:
            if utt:
                _, token_ids = self.tokenize_and_encode(utt)
            else:
                token_ids = [0]  # PAD
            hist_tokens.append(token_ids)
        
        # Encode intents
        hist_intent_ids = []
        for intent in history_intents:
            intent_id = self.mappings['intent2id'].get(intent, 0)
            hist_intent_ids.append(intent_id)
        
        # Encode dialog acts
        hist_da_ids = []
        for da in history_dialog_acts:
            da_id = self.mappings['da2id'].get(da, 0)
            hist_da_ids.append(da_id)
        
        # Convert to tensors
        # Pad all history token sequences to same length
        max_len = max(len(tokens) for tokens in hist_tokens)
        hist_tokens_padded = []
        for tokens in hist_tokens:
            padded = tokens + [0] * (max_len - len(tokens))
            hist_tokens_padded.append(padded)
        
        hist_tokens_tensor = torch.tensor(hist_tokens_padded, dtype=torch.long).unsqueeze(0)  # (1, W, L)
        hist_intents_tensor = torch.tensor(hist_intent_ids, dtype=torch.long).unsqueeze(0)  # (1, W)
        hist_das_tensor = torch.tensor(hist_da_ids, dtype=torch.long).unsqueeze(0)  # (1, W)
        
        return hist_tokens_tensor, hist_intents_tensor, hist_das_tensor
    
    @torch.no_grad()
    def predict(self, utterance: str, 
                history_utterances: List[str],
                history_intents: List[str],
                history_dialog_acts: List[str]) -> Dict:
        """
        Run inference on a single utterance
        Returns: dict with intent, slots, requested_slots, confidence
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize current utterance
        tokens, token_ids = self.tokenize_and_encode(utterance)
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Prepare history
        hist_tokens, hist_intents, hist_das = self.prepare_history(
            history_utterances, history_intents, history_dialog_acts
        )
        hist_tokens = hist_tokens.to(self.device)
        hist_intents = hist_intents.to(self.device)
        hist_das = hist_das.to(self.device)
        
        # Forward pass
        intent_logits, slot_logits, _, req_slot_logits = self.model(
            tokens_tensor, hist_tokens, hist_intents, hist_das
        )
        
        # Get predictions
        intent_pred = torch.argmax(intent_logits, dim=1).cpu().item()
        slot_preds = torch.argmax(slot_logits, dim=2).cpu().numpy()[0]
        
        # Get intent confidence
        intent_probs = torch.softmax(intent_logits, dim=1)
        intent_confidence = intent_probs[0, intent_pred].item()
        
        # Decode intent
        intent = self.mappings['id2intent'].get(intent_pred, 'NONE')
        
        # Extract slot values
        slot_values = enhanced_slot_value_extraction(
            slot_preds,
            tokens,
            utterance,
            self.mappings,
            use_utterance_fallback=True
        )
        
        # Extract requested slots
        requested_slots = extract_predicted_requested_slots(
            req_slot_logits[0],
            self.req_mappings['req_id2slot'],
            threshold=0.5
        )
        
        return {
            'intent': intent,
            'slots': slot_values,
            'requested_slots': requested_slots,
            'confidence': intent_confidence,
            'tokens': tokens
        }


# Global model manager instance
model_manager = ModelManager()
