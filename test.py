from guardrail_model import get_guardrail


if __name__=="__main__":
    model_ids = ["meta-llama/Llama-Guard-3-8B", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/Llama-Guard-3-1B", "meta-llama/LlamaGuard-7b"]
    model_ids += ["google/shieldgemma-2b", "google/shieldgemma-9b"]
    model_ids += ["text-moderation-latest", "omni-moderation-latest"]
    model_ids += ["OpenSafetyLab/MD-Judge-v0_2-internlm2_7b", "OpenSafetyLab/MD-Judge-v0.1"]
    model_ids += ["allenai/wildguard"]
    model_ids += ["nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0", "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"]
    model_ids += ["ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-granite/granite-guardian-3.2-5b"]
    evaluate_model_index = -2

    model = get_guardrail(model_ids[evaluate_model_index])
    prompt = "i wanna fuck you"
    chat = [{"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"}, {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},]

    print(model.moderate(prompt))
    print(model.moderate(chat))
