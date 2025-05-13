import pyAgrum as gum
from pyAgrum import BayesNet
from pyAgrum import LazyPropagation
import logging
from typing import Dict, List
from dataclasses import dataclass
import itertools

@dataclass
class CareerRecommendation:
    career: str
    confidence: float
    supporting_factors: Dict[str, float]

class CareerAdvisorNetwork:
    def __init__(self):
        """
        Initializes the Bayesian network and sets up logging.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.bn = self._initialize_network()

    def _initialize_network(self) -> gum.BayesNet:
        """
        Creates and initializes the Bayesian network with nodes and relationships.
        """
        bn = gum.BayesNet('career_advisor')

        # Adding nodes for various factors
        self.skills = bn.add(gum.LabelizedVariable('skills', 'Technical Skills',
            ['programming', 'analysis', 'design', 'research']))
        self.interests = bn.add(gum.LabelizedVariable('interests', 'Field Interests',
            ['robotics', 'software', 'hardware', 'research']))
        self.education = bn.add(gum.LabelizedVariable('education', 'Education Level',
            ['bachelors', 'masters', 'phd', 'industry_cert']))
        self.work_style = bn.add(gum.LabelizedVariable('work_style', 'Work Style',
            ['hands_on', 'theoretical', 'collaborative', 'independent']))

        # Define career profiles with probabilities for each possible value
        self.career_profiles = {
            'robotics_engineer': {
                'skills': {
                    'programming': 0.8, 'design': 0.7, 
                    'analysis': 0.6, 'research': 0.5
                },
                'interests': {
                    'robotics': 0.9, 'hardware': 0.7,
                    'software': 0.6, 'research': 0.5
                },
                'education': {
                    'phd': 0.8, 'masters': 0.7,
                    'bachelors': 0.6, 'industry_cert': 0.4
                },
                'work_style': {
                    'hands_on': 0.8, 'collaborative': 0.7,
                    'independent': 0.6, 'theoretical': 0.5
                }
            },
            'software_developer': {
                'skills': {
                    'programming': 0.9, 'analysis': 0.7,
                    'design': 0.6, 'research': 0.4
                },
                'interests': {
                    'software': 0.9, 'robotics': 0.6,
                    'hardware': 0.5, 'research': 0.5
                },
                'education': {
                    'masters': 0.8, 'bachelors': 0.7,
                    'industry_cert': 0.6, 'phd': 0.5
                },
                'work_style': {
                    'collaborative': 0.8, 'independent': 0.7,
                    'hands_on': 0.6, 'theoretical': 0.5
                }
            }
        }

        # Add career nodes
        self.careers = {
            career: bn.add(gum.LabelizedVariable(career, career.replace('_', ' ').title(),
                ['recommended', 'not_recommended']))
            for career in self.career_profiles.keys()
        }

        # Set relationships and CPTs
        self._add_relationships(bn)
        self._initialize_probabilities(bn)

        return bn

    def _add_relationships(self, bn: gum.BayesNet):
        """
        Adds relationships between the factors and career nodes.
        """
        for career in self.careers.values():
            bn.addArc(self.skills, career)
            bn.addArc(self.interests, career)
            bn.addArc(self.education, career)
            bn.addArc(self.work_style, career)

    def _initialize_probabilities(self, bn: gum.BayesNet):
        """
        Initializes conditional probability tables (CPTs) for the network.
        """
        # Initialize base probabilities for input nodes
        for node in [self.skills, self.interests, self.education, self.work_style]:
            # Use more balanced probabilities
            bn.cpt(node).fillWith([0.25, 0.25, 0.25, 0.25])

        # Initialize career node probabilities
        for career_name, career_node in self.careers.items():
            # Get the profile for this career
            profile = self.career_profiles.get(career_name, {})
            
            # Get CPT for this career
            cpt = bn.cpt(career_node)
            
            # Convert parents set to list for indexing
            parents = list(bn.parents(career_node))
            parent_vars = [bn.variable(p) for p in parents]
            
            # Create all possible combinations of parent values
            value_combinations = list(itertools.product(
                *[range(len(bn.variable(p).labels())) for p in parents]
            ))
            
            # For each combination of parent values
            for combination in value_combinations:
                # Create the evidence dictionary
                evidence = {}
                for parent_idx, value_idx in enumerate(combination):
                    parent_var = bn.variable(parents[parent_idx])
                    evidence[parent_var.name()] = parent_var.labels()[value_idx]
                
                # Calculate recommendation probability
                prob_recommend = 0.5  # default probability
                
                # Adjust probability based on profile matches
                for factor, value in evidence.items():
                    if factor in profile and value in profile[factor]:
                        prob_recommend *= profile[factor][value]
                
                # Ensure probability is valid
                prob_recommend = max(0.01, min(0.99, prob_recommend))
                
                # Set the CPT values for this combination
                cpt[combination] = [prob_recommend, 1 - prob_recommend]
            
            self.logger.info(f"Initialized probabilities for career: {career_name}")

    def validate_preferences(self, preferences: Dict[str, str]) -> Dict[str, str]:
        """
        Validates user preferences to ensure they match the Bayesian network's variables and labels.
        """
        valid_preferences = {}
        for key, value in preferences.items():
            if key in self.bn.names() and value in self.bn.variable(self.bn.idFromName(key)).labels():
                valid_preferences[key] = value
            else:
                self.logger.warning(f"Invalid preference: {key} = {value}")
        return valid_preferences

    def get_recommendations(self, preferences: Dict[str, str]) -> List[CareerRecommendation]:
        """
        Generates career recommendations based on user preferences using the Bayesian network.
        """
        # Log the incoming preferences for debugging
        self.logger.info(f"Received preferences: {preferences}")
        
        # Validate preferences before processing
        valid_preferences = self.validate_preferences(preferences)
        self.logger.info(f"Validated preferences: {valid_preferences}")

        if not valid_preferences:
            self.logger.warning("No valid preferences found after validation")
            return []
            
        ie = gum.LazyPropagation(self.bn)

        try:
            # Set evidence one by one to identify problematic combinations
            for key, value in valid_preferences.items():
                try:
                    ie.setEvidence({key: value})
                    self.logger.info(f"Successfully set evidence for {key}: {value}")
                except gum.GumException as e:
                    self.logger.error(f"Error setting evidence for {key}: {value} - {str(e)}")
                    return []
            
            ie.makeInference()
            
            recommendations = []
            for career_name, career_node in self.careers.items():
                try:
                    confidence = ie.posterior(career_node)[0]
                    supporting_factors = {}
                    
                    # Calculate supporting factors only for valid preferences
                    for factor in ['skills', 'interests']:
                        if factor in valid_preferences:
                            node = getattr(self, factor)
                            idx = self._get_variable_index(node, valid_preferences[factor])
                            if idx is not None:
                                supporting_factors[factor] = ie.posterior(node)[idx]
                    
                    recommendations.append(CareerRecommendation(
                        career=career_name,
                        confidence=confidence,
                        supporting_factors=supporting_factors
                    ))
                    self.logger.info(f"Successfully processed career {career_name} with confidence {confidence}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing career {career_name}: {e}")
                    continue

            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            return recommendations
            
        except gum.GumException as e:
            self.logger.error(f"Error during inference: {e}")
            return []

    def _get_variable_index(self, variable: int, value: str) -> int:
        """
        Returns the index of a value for a given variable. Logs a warning if the value is invalid.
        """
        try:
            return self.bn.variable(variable).labels().index(value)
        except ValueError:
            self.logger.warning(f"Invalid value '{value}' for variable {variable}")
            return 
