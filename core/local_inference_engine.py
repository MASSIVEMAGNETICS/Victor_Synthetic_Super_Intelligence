#!/usr/bin/env python3
"""
LocalInferenceEngine
Production-grade local LLM inference for Victor Synthetic Super Intelligence.

Primary: Ollama (fast, chat-native)
Fallback: llama.cpp (GGUF, pre-loaded for speed)

Merged from itz local AI improvements - May 2026
"""

import time
import psutil
from typing import Optional, List, Dict, Any

# === Backend Availability Checks (once at import) ===
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class LocalInferenceEngine:
    """
    Robust, high-performance local inference engine.
    
    Features:
    - Ollama first (recommended for speed & chat format)
    - Pre-loaded llama.cpp fallback (no reload penalty)
    - Consistent chat completion interface
    - Time logging + self-optimization hooks
    - Production error handling
    """

    def __init__(self, model: str = "llama3.2", n_ctx: int = 4096):
        self.model = model
        self.log: List[float] = []
        self.llm: Optional[Llama] = None
        self.n_ctx = n_ctx

        # Pre-load llama.cpp model once (critical for performance)
        if LLAMA_CPP_AVAILABLE:
            try:
                model_path = f"./{model}.gguf"
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=psutil.cpu_count(),
                    # n_gpu_layers=-1,  # Enable if CUDA available
                    verbose=False
                )
                print(f"[LocalInferenceEngine] ✅ llama.cpp pre-loaded: {model_path}")
            except Exception as e:
                print(f"[LocalInferenceEngine] ⚠️  llama.cpp preload failed: {e}")
                self.llm = None

    def infer(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Main inference method.
        
        Args:
            prompt: User prompt (used if messages=None)
            max_tokens: Max output tokens
            temperature: Sampling temperature
            messages: Optional chat history for multi-turn
        
        Returns:
            Generated text response
        """
        start = time.time()
        out = None
        last_error = None

        # Build messages if not provided
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # === Primary: Ollama ===
        if OLLAMA_AVAILABLE:
            try:
                r = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                )
                out = r["message"]["content"]
            except Exception as e:
                last_error = f"Ollama failed: {e}"

        # === Fallback: Pre-loaded llama.cpp ===
        if out is None and self.llm is not None:
            if last_error:
                print(f"{last_error} → falling back to llama.cpp")
            try:
                chat_response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                out = chat_response["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = f"llama.cpp fallback failed: {e}"

        if out is None:
            raise RuntimeError(
                f"[LocalInferenceEngine] No working backend. Last error: {last_error}"
            )

        # === Hooks (preserved from original) ===
        self.reroute()
        dur = time.time() - start
        self.log.append(dur)
        self.self_optimize(dur)

        return out.strip()

    def reroute(self):
        """Override or extend this method for custom routing logic."""
        pass

    def self_optimize(self, duration: float):
        """Override or extend for performance self-tuning."""
        if duration > 5.0:
            print(f"[LocalInferenceEngine] Slow inference ({duration:.2f}s) - consider model swap or quantization")

    def get_stats(self) -> Dict[str, Any]:
        """Return inference statistics."""
        if not self.log:
            return {"calls": 0, "avg_time": 0.0}
        return {
            "calls": len(self.log),
            "avg_time": sum(self.log) / len(self.log),
            "last_time": self.log[-1]
        }


if __name__ == "__main__":
    engine = LocalInferenceEngine(model="llama3.2")
    response = engine.infer("Explain the meaning of life in one sentence.")
    print("Response:", response)
    print("Stats:", engine.get_stats())
