"""
Complete inference pipeline
"""
from typing import Dict, List
from app.model_manager import model_manager
from app.session_manager import session_manager
from app.response_policy import response_policy
from app.schemas import get_domain_from_intent, SlotValue


class InferencePipeline:
    """
    Complete inference pipeline: NLU -> State Tracking -> Response Generation
    """
    
    def __init__(self):
        self.model = model_manager
        self.session_mgr = session_manager
        self.response_gen = response_policy
    
    def process_message(self, message: str, session_id: str) -> Dict:
        """
        Process user message through complete pipeline
        
        Returns:
            {
                'intent': str,
                'slots': List[SlotValue],
                'requested_slots': List[str],
                'response': Optional[str],
                'domain': str,
                'confidence': float,
                'accumulated_state': Dict[str, List[str]]
            }
        """
        # Get session context
        session = self.session_mgr.get_session(session_id)
        if not session:
            # Create new session
            session_id = self.session_mgr.create_session()
            session = self.session_mgr.get_session(session_id)
        
        # Get conversation history
        hist_utterances, hist_intents, hist_das = self.session_mgr.get_context_for_model(
            session_id, context_window=3
        )
        
        # Run NLU model
        predictions = self.model.predict(
            utterance=message,
            history_utterances=hist_utterances,
            history_intents=hist_intents,
            history_dialog_acts=hist_das
        )
        
        # Extract predictions
        intent = predictions['intent']
        turn_slots = predictions['slots']  # Dict[str, List[str]]
        requested_slots = predictions['requested_slots']
        confidence = predictions['confidence']
        
        # Update session and state tracker
        self.session_mgr.update_session(
            session_id=session_id,
            utterance=message,
            intent=intent,
            slots=turn_slots
        )
        
        # Get accumulated dialogue state
        accumulated_state = self.session_mgr.get_dialogue_state(session_id)
        
        # Determine domain
        domain = get_domain_from_intent(intent)
        if not domain:
            domain = "UNKNOWN"
        
        # Generate response (only for supported domains)
        response = None
        if domain in self.response_gen.supported_domains:
            response = self.response_gen.generate_response(
                domain=domain,
                intent=intent,
                current_slots=turn_slots,
                accumulated_state=accumulated_state,
                requested_slots=requested_slots
            )
        
        # Format slots for API response
        slots_formatted = []
        for slot_name, values in turn_slots.items():
            for value in values:
                slots_formatted.append(SlotValue(slot=slot_name, value=value))
        
        return {
            'session_id': session_id,
            'intent': intent,
            'slots': slots_formatted,
            'requested_slots': requested_slots,
            'response': response,
            'domain': domain,
            'confidence': confidence,
            'accumulated_state': accumulated_state
        }
    
    def reset_session(self, session_id: str):
        """Reset a session"""
        self.session_mgr.delete_session(session_id)


# Global inference pipeline
inference_pipeline = InferencePipeline()
