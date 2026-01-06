"""
Rule-based response generation for supported domains
"""
from typing import Dict, List, Optional
from app.schemas import DOMAIN_SCHEMAS


class ResponsePolicy:
    """
    Generates natural language responses for supported domains
    """
    
    def __init__(self, supported_domains: List[str]):
        self.supported_domains = supported_domains
        self.domain_schemas = DOMAIN_SCHEMAS
    
    def generate_response(self, domain: str, intent: str, 
                         current_slots: Dict[str, List[str]],
                         accumulated_state: Dict[str, List[str]],
                         requested_slots: List[str]) -> Optional[str]:
        """
        Generate response for supported domains
        Returns None if domain not supported
        """
        if domain not in self.supported_domains:
            return None
        
        if domain == "Buses_2":
            return self._generate_bus_response(intent, current_slots, accumulated_state, requested_slots)
        elif domain == "Movies_1":
            return self._generate_movie_response(intent, current_slots, accumulated_state, requested_slots)
        
        return None
    
    def _generate_bus_response(self, intent: str, current_slots: Dict[str, List[str]],
                               accumulated_state: Dict[str, List[str]],
                               requested_slots: List[str]) -> str:
        """Generate response for Buses_2 domain"""
        
        if "FindBus" in intent:
            # Check what information we have
            origin = accumulated_state.get('origin', [None])[0]
            destination = accumulated_state.get('destination', [None])[0]
            date = accumulated_state.get('departure_date', [None])[0]
            
            # Missing required slots
            missing = []
            if not origin:
                missing.append("origin city")
            if not destination:
                missing.append("destination city")
            if not date:
                missing.append("departure date")
            
            if missing:
                return f"To find buses, I need to know the {', '.join(missing)}. Could you provide that information?"
            
            # Has all required info
            group_size = accumulated_state.get('group_size', ['1'])[0]
            fare_type = accumulated_state.get('fare_type', ['Economy'])[0]
            
            return (f"I'm searching for {fare_type} buses from {origin} to {destination} "
                   f"on {date} for {group_size} passenger(s). "
                   f"I found several options. Would you like me to show you the available buses?")
        
        elif "BuyBusTicket" in intent:
            origin = accumulated_state.get('origin', [None])[0]
            destination = accumulated_state.get('destination', [None])[0]
            date = accumulated_state.get('departure_date', [None])[0]
            time = accumulated_state.get('departure_time', [None])[0]
            group_size = accumulated_state.get('group_size', [None])[0]
            
            missing = []
            if not origin or not destination:
                missing.append("route information")
            if not date:
                missing.append("departure date")
            if not time:
                missing.append("departure time")
            if not group_size:
                missing.append("number of passengers")
            
            if missing:
                return f"To book tickets, I need: {', '.join(missing)}. What would you like to provide?"
            
            fare_type = accumulated_state.get('fare_type', ['Economy'])[0]
            
            return (f"I'll book {group_size} {fare_type} ticket(s) from {origin} to {destination} "
                   f"on {date} at {time}. Please confirm to proceed with the booking.")
        
        # Handle requested slots
        if requested_slots:
            slots_desc = []
            for slot in requested_slots:
                if slot in accumulated_state:
                    value = accumulated_state[slot][0]
                    slots_desc.append(f"{slot.replace('_', ' ')}: {value}")
            
            if slots_desc:
                return f"Here's the information you requested: {', '.join(slots_desc)}"
        
        return "I can help you find buses or book tickets. What would you like to do?"
    
    def _generate_movie_response(self, intent: str, current_slots: Dict[str, List[str]],
                                accumulated_state: Dict[str, List[str]],
                                requested_slots: List[str]) -> str:
        """Generate response for Movies_1 domain"""
        
        if "FindMovies" in intent:
            location = accumulated_state.get('location', [None])[0]
            
            if not location:
                return "Which city would you like to find movies in?"
            
            genre = accumulated_state.get('genre')
            show_type = accumulated_state.get('show_type')
            theater = accumulated_state.get('theater_name')
            
            filters = []
            if genre and genre[0] not in ['dontcare', 'any']:
                filters.append(f"{genre[0]} movies")
            if show_type and show_type[0] not in ['dontcare', 'regular']:
                filters.append(f"{show_type[0]} shows")
            if theater:
                filters.append(f"at {theater[0]}")
            
            filter_text = " ".join(filters) if filters else "movies"
            return f"I'm searching for {filter_text} in {location}. I found several options available."
        
        elif "GetTimesForMovie" in intent:
            movie = accumulated_state.get('movie_name', [None])[0]
            location = accumulated_state.get('location', [None])[0]
            date = accumulated_state.get('show_date', [None])[0]
            
            missing = []
            if not movie:
                missing.append("movie name")
            if not location:
                missing.append("location")
            if not date:
                missing.append("date")
            
            if missing:
                return f"To check showtimes, I need: {', '.join(missing)}."
            
            show_type = accumulated_state.get('show_type', ['regular'])[0]
            theater = accumulated_state.get('theater_name')
            
            theater_text = f" at {theater[0]}" if theater else ""
            return (f"Let me find {show_type} showtimes for '{movie}' in {location} "
                   f"on {date}{theater_text}.")
        
        elif "BuyMovieTickets" in intent:
            movie = accumulated_state.get('movie_name', [None])[0]
            location = accumulated_state.get('location', [None])[0]
            date = accumulated_state.get('show_date', [None])[0]
            time = accumulated_state.get('show_time', [None])[0]
            tickets = accumulated_state.get('number_of_tickets', [None])[0]
            show_type = accumulated_state.get('show_type', [None])[0]
            
            missing = []
            if not movie:
                missing.append("movie name")
            if not location:
                missing.append("location")
            if not date:
                missing.append("show date")
            if not time:
                missing.append("show time")
            if not tickets:
                missing.append("number of tickets")
            if not show_type:
                missing.append("show type")
            
            if missing:
                return f"To book tickets, I need: {', '.join(missing)}."
            
            return (f"I'll book {tickets} ticket(s) for the {show_type} show of '{movie}' "
                   f"in {location} on {date} at {time}. Please confirm to proceed.")
        
        # Handle requested slots
        if requested_slots:
            slots_desc = []
            for slot in requested_slots:
                if slot in accumulated_state:
                    value = accumulated_state[slot][0]
                    slots_desc.append(f"{slot.replace('_', ' ')}: {value}")
            
            if slots_desc:
                return f"Here's what you asked about: {', '.join(slots_desc)}"
        
        return "I can help you find movies, check showtimes, or book tickets. What would you like to do?"
    
    def get_missing_slots(self, domain: str, intent: str, 
                         current_state: Dict[str, List[str]]) -> List[str]:
        """Get list of missing required slots for an intent"""
        if domain not in self.domain_schemas:
            return []
        
        # Extract intent name without domain prefix
        intent_name = intent.split('_', 2)[-1] if '_' in intent else intent
        
        intent_config = self.domain_schemas[domain]['intents'].get(intent_name)
        if not intent_config:
            return []
        
        required_slots = intent_config.get('required_slots', [])
        missing = [slot for slot in required_slots if slot not in current_state or not current_state[slot]]
        
        return missing


# Global response policy instance
response_policy = ResponsePolicy(supported_domains=["Buses_2", "Movies_1"])
