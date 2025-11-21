#!/usr/bin/env python3
"""
CommandRouter - maps recognized intent + slots into VictorHub Task objects and invokes Hub.execute_task.
This module expects a VictorHub instance with execute_task(task) method and a SkillRegistry.
"""
from typing import Dict, Any
from victor_hub.victor_boot import Task

# Simple intent->task_type mapping (extendable)
INTENT_TASK_MAP = {
    "summarize": "summarization",
    "extract_entities": "ner",
    "sentiment_analysis": "sentiment",
    "keyword_extraction": "keyword_extraction",
    "pos_tagging": "pos_tagging",
    "dependency_parsing": "dependency_parsing",
    "run_command": "command",  # custom handler
    "search": "search",
    "create_note": "create_note"
}

class CommandRouter:
    def __init__(self, hub):
        self.hub = hub

    def route(self, intent_output: Dict[str, Any], user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build a Task from intent output and dispatch to hub.execute_task.
        intent_output: {"intent": str, "confidence": float, "slots": {...}}
        Returns hub.execute_task() result dict.
        """
        intent = intent_output.get("intent")
        slots = intent_output.get("slots", {})
        task_type = INTENT_TASK_MAP.get(intent, "nlp")

        # Build Task
        task = Task(
            id=f"intent-{intent}",
            type=task_type,
            description=f"Auto-generated task for intent: {intent}",
            inputs={"text": user_context.get("text") if user_context else slots.get("quoted") or slots.get("noun_chunks") or ""}
        )

        # If special 'command' intent, include slots so command skill can parse
        task.inputs["slots"] = slots
        # Dispatch
        result = self.hub.execute_task(task)
        
        # Convert Result object to dict for compatibility
        return {
            "task_id": result.task_id,
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata
        }
