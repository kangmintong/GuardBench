from .llamaguard import LlamaGuard
from .shieldgemma import SheildGemma
from .openai_mod import OpenAIMod
from .mdjudge import MDJudge
from .wildguard import WildGuard
from .llmjudge import LLMJudge

def get_guardrail(model):
    if "llama" in model.lower():
        return LlamaGuard(model)
    elif "gemma" in model.lower():
        return SheildGemma(model)
    elif "moderation" in model.lower():
        return OpenAIMod(model)
    elif "judge" in model.lower():
        return MDJudge(model)
    elif "wildguard" in model.lower():
        return WildGuard(model)
    elif "llmjudge" in model.lower():
        return LLMJudge(model)
    else:
        raise ValueError(f"Unsupported guardrail model {model}")
