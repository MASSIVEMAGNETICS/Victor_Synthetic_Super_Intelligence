"""
Victor Hub Skills Package
Contains all skill implementations for the Victor Hub
"""

__version__ = "1.0.0"

# Import all skills for easy registration
from .echo_skill import EchoSkill
from .content_generator import ContentGeneratorSkill
from .research_agent import ResearchAgentSkill
from .nlp_skill import AdvancedNLPSkill
from .intent_skill import IntentSkill

__all__ = [
    "EchoSkill",
    "ContentGeneratorSkill", 
    "ResearchAgentSkill",
    "AdvancedNLPSkill",
    "IntentSkill"
]
