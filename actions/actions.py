from typing import Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
from .career_advisor import CareerAdvisorNetwork, CareerRecommendation  # Import the network and data class

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the career advisor network
career_advisor = CareerAdvisorNetwork()

class ActionCareerRecommendation(Action):
    def name(self) -> str:
        return "action_career_recommendation"
    
    def normalize_work_style(self, work_style: str) -> str:
        """Normalize work style input to match Bayesian network values."""
        style_mapping = {
            'collaboratively': 'collaborative',
            'independently': 'independent',
            'hands on': 'hands_on',
            'theoretical': 'theoretical'
        }
        return style_mapping.get(work_style.lower(), work_style)

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        # Extract user preferences from tracker slots
        preferences = {
            "skills": tracker.get_slot("skills"),
            "interests": tracker.get_slot("interests"),
            "education": tracker.get_slot("education"),
            "work_style": tracker.get_slot("work_style")
        }

        # Filter out empty or None preferences and normalize work_style
        filtered_preferences = {}
        for key, value in preferences.items():
            if value:
                if key == 'work_style':
                    value = self.normalize_work_style(value)
                filtered_preferences[key] = value
                
        print(filtered_preferences)

        if not filtered_preferences:
            dispatcher.utter_message(text="I couldn't find enough information about your preferences. Please provide more details.")
            return []

        try:
            recommendations = career_advisor.get_recommendations(filtered_preferences)

            if recommendations:
                response = "Based on your preferences, here are the top career recommendations:\n"
                for idx, rec in enumerate(recommendations[:3], start=1):
                    confidence_pct = rec.confidence * 100
                    response += f"{idx}. {rec.career.replace('_', ' ').title()} - Confidence: {confidence_pct:.1f}%\n"
                    
                    # Add supporting factors explanation
                    response += "   Supporting factors:\n"
                    for factor, value in rec.supporting_factors.items():
                        response += f"   - {factor.title()}: {value:.1%}\n"
                
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text="I couldn't generate any career recommendations. Please provide more specific preferences.")
        except Exception as e:
            logger.error(f"Error during career recommendation: {e}")
            dispatcher.utter_message(text="An error occurred while generating recommendations. Please try again later.")
        
        return []