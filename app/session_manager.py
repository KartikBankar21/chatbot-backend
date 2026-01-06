"""
Session and dialogue state management
"""
import uuid
import time
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock

from models.dialogue_state_tracker_two import DialogueStateTracker


class SessionManager:
    """
    Manages user sessions and dialogue states
    """
    
    def __init__(self, timeout: int = 3600, max_history: int = 10):
        self.sessions: Dict[str, dict] = {}
        self.state_tracker = DialogueStateTracker()
        self.timeout = timeout
        self.max_history = max_history
        self.lock = Lock()
    
    def create_session(self) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        
        with self.lock:
            self.sessions[session_id] = {
                'created_at': time.time(),
                'last_access': time.time(),
                'turn_count': 0,
                'history': [],  # List of (utterance, intent, slots) tuples
                'context_embeddings': []  # For model context
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check timeout
            if time.time() - session['last_access'] > self.timeout:
                self.delete_session(session_id)
                return None
            
            session['last_access'] = time.time()
            return session
    
    def update_session(self, session_id: str, utterance: str, 
                      intent: str, slots: Dict[str, List[str]],
                      context_embedding: Optional[list] = None):
        """Update session with new turn"""
        with self.lock:
            if session_id not in self.sessions:
                return
            
            session = self.sessions[session_id]
            session['turn_count'] += 1
            session['last_access'] = time.time()
            
            # Add to history
            session['history'].append({
                'utterance': utterance,
                'intent': intent,
                'slots': slots
            })
            
            # Keep only recent history
            if len(session['history']) > self.max_history:
                session['history'] = session['history'][-self.max_history:]
            
            # Update context embeddings
            if context_embedding is not None:
                session['context_embeddings'].append(context_embedding)
                if len(session['context_embeddings']) > self.max_history:
                    session['context_embeddings'] = session['context_embeddings'][-self.max_history:]
            
            # Update dialogue state
            self.state_tracker.update(session_id, slots)
    
    def get_dialogue_state(self, session_id: str) -> Dict[str, List[str]]:
        """Get accumulated dialogue state"""
        return self.state_tracker.get_state(session_id)
    
    def get_context_for_model(self, session_id: str, context_window: int = 3):
        """
        Get recent context for model input
        Returns: (history_utterances, history_intents, history_dialog_acts)
        """
        session = self.get_session(session_id)
        if not session:
            return [], [], []
        
        history = session['history'][-context_window:]
        
        utterances = [h['utterance'] for h in history]
        intents = [h['intent'] for h in history]
        # For dialog acts, we'll use a simple mapping (you can enhance this)
        dialog_acts = ['INFORM' for _ in history]  # Simplified
        
        return utterances, intents, dialog_acts
    
    def delete_session(self, session_id: str):
        """Delete session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            self.state_tracker.reset_dialogue(session_id)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired = []
        
        with self.lock:
            for sid, session in self.sessions.items():
                if current_time - session['last_access'] > self.timeout:
                    expired.append(sid)
        
        for sid in expired:
            self.delete_session(sid)
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get session information"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        state = self.get_dialogue_state(session_id)
        last_intent = session['history'][-1]['intent'] if session['history'] else "NONE"
        
        return {
            'session_id': session_id,
            'turn_count': session['turn_count'],
            'current_state': state,
            'last_intent': last_intent
        }


# Global session manager instance
session_manager = SessionManager()
