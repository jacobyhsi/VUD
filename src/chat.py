import requests
import math

def chat(message: str, label_keys, seed: int, model: str ="Qwen/Qwen2.5-14B", port: str = "8000", ip: str = "localhost"):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": message,
        "temperature": 1.0,
        "max_tokens": 5,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data).json()
    text_output = response["choices"][0]["text"]
    # print("\n### Full Output ###\n" + text_output)

    logprobs_list = response["choices"][0].get("logprobs", {}).get("top_logprobs", [])
    tokens = response["choices"][0].get("logprobs", {}).get("tokens", [])

    label_logprobs = {}
    for i, token in enumerate(tokens):
        stripped_token = token.strip()
        if stripped_token in label_keys and i < len(logprobs_list):
            full_logprobs = logprobs_list[i]
            label_logprobs = {token_option: logprob for token_option, logprob in full_logprobs.items() if token_option.strip() in label_keys}
            break  # Stop after finding the first valid label
    
    # Normalization
    exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}

    # print("\n### Normalized Probabilities ###\n" + str(normalized_probs))

    return text_output, normalized_probs

def chat_response_only(message: str, seed: int, max_tokens: int=10, temperature: float=1.0, model: str="Qwen/Qwen2.5-14B", port: str="8000", ip: str="localhost"):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": message,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data).json()
    text_output = response["choices"][0]["text"]
    
    return text_output

### QA ###
def chat_perturb(message: str, seed: int, max_tokens: int=512, model: str="Qwen/Qwen2.5-14B-Instruct"):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": message,
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data).json()
    text_output = response["choices"][0]["text"]
    
    return text_output

def chat_qa(message: str, label_keys, seed: int, model: str="Qwen/Qwen2.5-14B-Instruct"):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": message,
        "temperature": 1.0,
        "max_tokens": 10,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    text_output = response_json["choices"][0]["text"]

    logprobs_list = response_json["choices"][0].get("logprobs", {}).get("top_logprobs", [])

    label_logprobs = {}

    # Check all tokens, not just first
    for token_probs in logprobs_list:
        for token_option, logprob in token_probs.items():
            stripped_token = token_option.strip()
            if stripped_token in label_keys:
                # Keep max logprob if multiple occurrences
                if (stripped_token not in label_logprobs) or (logprob > label_logprobs[stripped_token]):
                    label_logprobs[stripped_token] = logprob

    # Normalize to probabilities
    exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}

    return text_output, normalized_probs
