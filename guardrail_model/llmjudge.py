from .base import BaseGuardrailModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import List
from .utils import ModerationResult

class LLMJudge(BaseGuardrailModel):
    def __init__(self, model_id = "allenai/wildguard", device = "cuda"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

    def moderate(self, prompt: str or List[dict]) -> dict:
        moderation_result = ModerationResult(prompt=prompt, model=self.model_id)
        return moderation_result