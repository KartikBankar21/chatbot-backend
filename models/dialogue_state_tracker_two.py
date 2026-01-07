"""
Dialogue State Tracker
Accumulates slot values across turns to compute Joint/Average Goal Accuracy
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import torch


class DialogueStateTracker:
    """
    Tracks dialogue state across multiple turns.
    Accumulates slot-value pairs and only updates when new values are detected.
    """
    
    def __init__(self):
        # Track state per dialogue: {dialogue_id: {slot_name: value}}
        self.states = defaultdict(dict)
    
    def update(self, dialogue_id: str, new_slot_values: Dict[str, List[str]]):
        """
        Update dialogue state with new slot values from current turn.
        
        Args:
            dialogue_id: Unique dialogue identifier
            new_slot_values: Dict mapping slot_name -> [values] from current turn
        """
        if dialogue_id not in self.states:
            self.states[dialogue_id] = {}
        
        # Update state with new values
        for slot, values in new_slot_values.items():
            if values and len(values) > 0:
                # Take the first value (or you could aggregate differently)
                value = values[0]
                
                # Only update if non-empty and not placeholder
                if value and value.strip().lower() not in ['', 'none', 'null', '<unk>']:
                    self.states[dialogue_id][slot] = value
    
    def get_state(self, dialogue_id: str) -> Dict[str, List[str]]:
        """
        Get current accumulated state for a dialogue.
        Returns in SGD format: {slot_name: [value]}
        """
        state = self.states.get(dialogue_id, {})
        # Convert to SGD format (values as lists)
        return {slot: [value] for slot, value in state.items()}
    
    def reset_dialogue(self, dialogue_id: str):
        """Reset state for a specific dialogue"""
        if dialogue_id in self.states:
            del self.states[dialogue_id]
    
    def reset_all(self):
        """Reset all dialogue states"""
        self.states.clear()


def extract_predicted_requested_slots(
    req_slot_logits: torch.Tensor,
    req_id2slot: Dict[int, str],
    threshold: float = 0.5
) -> List[str]:
    """
    Extract predicted requested slots from model output.
    
    Args:
        req_slot_logits: (batch, num_req_slots) or (num_req_slots,)
        req_id2slot: Mapping from ID to slot name
        threshold: Probability threshold for considering slot as requested
        
    Returns:
        List of requested slot names
    """
    # Handle batch dimension
    if req_slot_logits.dim() == 2:
        req_slot_logits = req_slot_logits[0]  # Take first in batch
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(req_slot_logits).detach().cpu().numpy()
    
    # Get slots above threshold
    requested_slots = []
    for slot_id, prob in enumerate(probs):
        if prob >= threshold:
            slot_name = req_id2slot.get(slot_id)
            if slot_name:
                requested_slots.append(slot_name)
    
    return requested_slots


def enhanced_slot_value_extraction(
    slot_pred_ids,
    tokens_text,
    utterance_text,
    mappings,
    use_utterance_fallback=True
):
    """
    Enhanced slot value extraction with better handling.
    
    Args:
        slot_pred_ids: Predicted slot IDs (BIO tags)
        tokens_text: List of tokens
        utterance_text: Original utterance string
        mappings: id2slot mapping
        use_utterance_fallback: If True, extract from original utterance for better matches
        
    Returns:
        Dict[str, List[str]] - Extracted slot values
    """
    slot_values = {}
    current_slot = None
    current_value_tokens = []
    current_char_spans = []
    
    # Track character positions for fallback
    char_pos = 0
    token_to_chars = []
    
    for token in tokens_text:
        # Find token position in utterance (handling tokenization artifacts)
        token_clean = token.replace('##', '')  # BERT subword handling
        start = utterance_text.find(token_clean, char_pos)
        if start != -1:
            end = start + len(token_clean)
            token_to_chars.append((start, end))
            char_pos = end
        else:
            token_to_chars.append((None, None))
    
    # Extract slots from BIO predictions
    for i, (token, slot_id) in enumerate(zip(tokens_text, slot_pred_ids)):
        if slot_id < 0 or slot_id >= len(mappings['id2slot']):
            continue
        
        slot_tag = mappings['id2slot'][slot_id]
        
        if slot_tag == 'O':
            # Save previous slot
            if current_slot and current_value_tokens:
                if use_utterance_fallback and current_char_spans:
                    # FIXED: Check if there are valid spans before using min/max
                    valid_spans = [(s, e) for s, e in current_char_spans if s is not None and e is not None]
                    if valid_spans:
                        start = min(s for s, e in valid_spans)
                        end = max(e for s, e in valid_spans)
                        value = utterance_text[start:end].strip()
                    else:
                        # Fallback to token concatenation
                        value = ' '.join(current_value_tokens).replace(' ##', '')
                else:
                    # Fallback to token concatenation
                    value = ' '.join(current_value_tokens).replace(' ##', '')
                
                if current_slot not in slot_values:
                    slot_values[current_slot] = []
                if value:  # Only add non-empty values
                    slot_values[current_slot].append(value)
            
            current_slot = None
            current_value_tokens = []
            current_char_spans = []
        
        elif slot_tag.startswith('B-'):
            # Save previous and start new
            if current_slot and current_value_tokens:
                if use_utterance_fallback and current_char_spans:
                    valid_spans = [(s, e) for s, e in current_char_spans if s is not None and e is not None]
                    if valid_spans:
                        start = min(s for s, e in valid_spans)
                        end = max(e for s, e in valid_spans)
                        value = utterance_text[start:end].strip()
                    else:
                        value = ' '.join(current_value_tokens).replace(' ##', '')
                else:
                    value = ' '.join(current_value_tokens).replace(' ##', '')
                
                if current_slot not in slot_values:
                    slot_values[current_slot] = []
                if value:  # Only add non-empty values
                    slot_values[current_slot].append(value)
            
            current_slot = slot_tag[2:]
            current_value_tokens = [token]
            current_char_spans = [token_to_chars[i]]
        
        elif slot_tag.startswith('I-'):
            if current_slot:
                current_value_tokens.append(token)
                current_char_spans.append(token_to_chars[i])
    
    # Save final slot
    if current_slot and current_value_tokens:
        if use_utterance_fallback and current_char_spans:
            valid_spans = [(s, e) for s, e in current_char_spans if s is not None and e is not None]
            if valid_spans:
                start = min(s for s, e in valid_spans)
                end = max(e for s, e in valid_spans)
                value = utterance_text[start:end].strip()
            else:
                value = ' '.join(current_value_tokens).replace(' ##', '')
        else:
            value = ' '.join(current_value_tokens).replace(' ##', '')
        
        if current_slot not in slot_values:
            slot_values[current_slot] = []
        if value:  # Only add non-empty values
            slot_values[current_slot].append(value)
    
    return slot_values


# ============================================================================
# Integration Functions
# ============================================================================

def process_predictions_with_state_tracking(
    intent_preds,
    slot_preds,
    req_slot_preds,
    tokens_text,
    utterance_text,
    dialogue_ids,
    mappings,
    state_tracker: DialogueStateTracker
) -> List[Tuple[str, Dict[str, List[str]], List[str]]]:
    """
    Process model predictions with dialogue state tracking.
    
    Args:
        intent_preds: Intent predictions (batch,)
        slot_preds: Slot predictions (batch, seq_len)
        req_slot_preds: Requested slot logits (batch, num_req_slots)
        tokens_text: List of token lists
        utterance_text: List of utterance strings
        dialogue_ids: List of dialogue IDs
        mappings: Mapping dictionaries
        state_tracker: DialogueStateTracker instance
        
    Returns:
        List of (intent, accumulated_slot_values, requested_slots) for each sample
    """
    batch_size = len(dialogue_ids)
    results = []
    
    for i in range(batch_size):
        # 1. Intent
        intent_id = intent_preds[i]
        if intent_id < len(mappings['id2intent']):
            intent = mappings['id2intent'][intent_id]
        else:
            intent = 'NONE'
        
        # 2. Extract slot values from current turn
        turn_slot_values = enhanced_slot_value_extraction(
            slot_preds[i],
            tokens_text[i],
            utterance_text[i],
            mappings,
            use_utterance_fallback=True
        )
        
        # 3. Update state tracker
        dialogue_id = dialogue_ids[i]
        state_tracker.update(dialogue_id, turn_slot_values)
        
        # 4. Get accumulated state
        accumulated_state = state_tracker.get_state(dialogue_id)
        
        # 5. Extract requested slots
        requested_slots = extract_predicted_requested_slots(
            req_slot_preds[i],
            mappings['req_id2slot'],
            threshold=0.5
        )
        
        results.append((intent, accumulated_state, requested_slots))
    
    return results


# ============================================================================
# Evaluation with State Tracking
# ============================================================================

def evaluate_with_state_tracking(
    model,
    dataloader,
    mappings,
    schemas,
    device='cuda'
):
    """
    Evaluate model with dialogue state tracking for accurate SGD metrics.
    
    Returns:
        Dictionary with SGD metrics
    """
    from sgd_evaluation import SGDMetricsTracker
    from sgd_fixed_integration import (
        extract_ground_truth_from_your_metadata,
        convert_schema_to_google_format
    )
    
    model.eval()
    
    # Initialize
    state_tracker = DialogueStateTracker()
    schemas_dict = {}
    sgd_data = []
    
    with torch.no_grad():
        for batch in dataloader:
            tokens, intents, slots, hist_tok, hist_int, hist_da, metadata = batch
            
            tokens = tokens.to(device)
            hist_tok = hist_tok.to(device)
            hist_int = hist_int.to(device)
            hist_da = hist_da.to(device)
            
            # Forward pass
            intent_logits, slot_logits, _, req_slot_logits = model(
                tokens, hist_tok, hist_int, hist_da
            )
            
            # Get predictions
            intent_preds = torch.argmax(intent_logits, dim=1).cpu().numpy()
            slot_preds = torch.argmax(slot_logits, dim=2).cpu().numpy()
            
            # Process with state tracking
            batch_size = tokens.size(0)
            
            for i in range(batch_size):
                try:
                    # Ground truth
                    service_name, intent_ref, slot_values_ref, requested_slots_ref, google_schema = \
                        extract_ground_truth_from_your_metadata({
                            'raw_frame': metadata['raw_frame'][i],
                            'service_schema': metadata['service_schema'][i]
                        })
                    
                    if service_name not in schemas_dict:
                        schemas_dict[service_name] = google_schema
                    
                    # Predictions with state tracking
                    dialogue_id = metadata['dialogue_id'][i]
                    
                    # Extract turn slot values
                    turn_slot_values = enhanced_slot_value_extraction(
                        slot_preds[i],
                        metadata['tokens_text'][i],
                        metadata['utterance_text'][i],
                        mappings
                    )
                    
                    # Update state
                    state_tracker.update(dialogue_id, turn_slot_values)
                    
                    # Get accumulated state (THIS IS KEY!)
                    slot_values_hyp = state_tracker.get_state(dialogue_id)
                    
                    # Intent
                    intent_hyp = mappings['id2intent'][intent_preds[i]]
                    
                    # Requested slots
                    requested_slots_hyp = extract_predicted_requested_slots(
                        req_slot_logits[i],
                        mappings['req_id2slot']
                    )
                    
                    sgd_data.append({
                        'service_name': service_name,
                        'intent_ref': intent_ref,
                        'intent_hyp': intent_hyp,
                        'slot_values_ref': slot_values_ref,
                        'slot_values_hyp': slot_values_hyp,
                        'requested_slots_ref': requested_slots_ref,
                        'requested_slots_hyp': requested_slots_hyp
                    })
                    
                except Exception as e:
                    print(f"Warning: Failed to process sample: {e}")
    
    # Compute metrics
    if sgd_data and schemas_dict:
        from sgd_evaluation import (
            ACTIVE_INTENT_ACCURACY, REQUESTED_SLOTS_F1,
            AVERAGE_GOAL_ACCURACY, JOINT_GOAL_ACCURACY
        )
        
        tracker = SGDMetricsTracker(schemas_dict)
        
        for item in sgd_data:
            tracker.update(**item)
        
        results = tracker.compute()
        
        return {
            'active_intent_acc': results[ACTIVE_INTENT_ACCURACY],
            'requested_slot_f1': results[REQUESTED_SLOTS_F1],
            'avg_goal_acc': results[AVERAGE_GOAL_ACCURACY],
            'joint_goal_acc': results[JOINT_GOAL_ACCURACY]
        }
    
    return {
        'active_intent_acc': 0.0,
        'requested_slot_f1': 0.0,
        'avg_goal_acc': 0.0,
        'joint_goal_acc': 0.0
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    # During evaluation:
    sgd_metrics = evaluate_with_state_tracking(
        model=enhanced_model,
        dataloader=val_loader,
        mappings=mappings,
        schemas=all_schemas,
        device='cuda'
    )
    
    print(f"With State Tracking:")
    print(f"  Avg Goal: {sgd_metrics['avg_goal_acc']:.4f}")
    print(f"  Joint Goal: {sgd_metrics['joint_goal_acc']:.4f}")
    """
    pass
