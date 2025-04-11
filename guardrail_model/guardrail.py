from .llamaguard import LlamaGuard
from .shieldgemma import SheildGemma
from .openai_mod import OpenAIMod
from .mdjudge import MDJudge
from .wildguard import WildGuard
from .llmjudge import LLMJudge
from .aegis import Aegis



def get_guardrail(model, cache_dir=None):
    if "llama" in model.lower() and "aegis" not in model.lower():
        return LlamaGuard(model, cache_dir=cache_dir)
    elif "gemma" in model.lower():
        return SheildGemma(model, cache_dir=cache_dir)
    elif "moderation" in model.lower():
        return OpenAIMod(model)
    elif "judge" in model.lower():
        return MDJudge(model, cache_dir=cache_dir)
    elif "wildguard" in model.lower():
        return WildGuard(model, cache_dir=cache_dir)
    elif "llmjudge" in model.lower():
        return LLMJudge(model)
    elif "aegis" in model.lower():
        return Aegis(model)
    elif "llmjudge" in model.lower():
        return LLMJudge(model, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported guardrail model {model}")
