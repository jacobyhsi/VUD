import requests
import math
import time
from openai import OpenAI, types
from src.llada_backend import generate_text, is_llada_model, score_labels

OPENAI_API_KEY = "ADD_API_KEY_HERE"  # Replace with your OpenAI API key

def _is_diffusion_gemma(model: str) -> bool:
    return "diffusiongemma" in model.lower()

def _parse_local_response(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Local model server returned non-JSON response "
            f"(HTTP {response.status_code}): {response.text[:500]}"
        ) from exc

    if not response.ok:
        error = payload.get("error", {})
        message = error.get("message", payload)
        raise RuntimeError(
            f"Local model server request failed (HTTP {response.status_code}): "
            f"{message}"
        )
    if not payload.get("choices"):
        raise RuntimeError(f"Local model server response has no choices: {payload}")

    return payload

def _post_local_json(url: str, headers: dict, data: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=300)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Local model server request failed after {max_retries} attempts: {exc}"
                ) from exc
        else:
            if response.status_code < 500 or attempt == max_retries - 1:
                return _parse_local_response(response)

        delay = 2 ** attempt
        print(
            f"Local model server request failed; retrying in {delay}s "
            f"({attempt + 1}/{max_retries})"
        )
        time.sleep(delay)

    raise RuntimeError("Local model server request failed without a response.")

def chat(message: str, label_keys, seed: int, temperature: float=1.0, model: str ="Qwen/Qwen2.5-14B", port: str = "8000", ip: str = "localhost", is_local_client: bool | int = True):
    if is_local_client and is_llada_model(model):
        return score_labels(
            message,
            label_keys,
            seed=seed,
            temperature=temperature,
            model_name=model,
        )

    if is_local_client:
        url = f"http://{ip}:{port}/v1/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": message,
            "max_tokens": 5,
            "logprobs": 10,
        }
        # Diffusion models reject temperature and seed.
        if not _is_diffusion_gemma(model):
            data["temperature"] = temperature
            data["seed"] = seed

        response = _post_local_json(url, headers, data)

        text_output = response["choices"][0]["text"]
        
        logprobs_list = response["choices"][0].get("logprobs", {}).get("top_logprobs", [])
        tokens = response["choices"][0].get("logprobs", {}).get("tokens", [])
    else:
        model = "gpt-4.1-nano-2025-04-14" # Default model for OpenAI API
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        response: types.Completion = client.completions.create(
            model=model,
            prompt=message,
            max_tokens=5,
            temperature=temperature,
            logprobs=10,
            seed=seed,
        )
        
        text_output = response.choices[0].text
        
        logprobs_list = response.choices[0].logprobs.top_logprobs
        tokens = response.choices[0].logprobs.tokens
    
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

def chat_response_only(message: str, seed: int, max_tokens: int=10, temperature: float=1.0, model: str="Qwen/Qwen2.5-14B", port: str="8000", ip: str="localhost", is_local_client: bool | int = True):
    if is_local_client and is_llada_model(model):
        return generate_text(
            message,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            model_name=model,
        )

    if is_local_client:
        headers = {"Content-Type": "application/json"}
        if _is_diffusion_gemma(model):
            # Use raw prompt continuation to avoid the chat-template thought channel.
            # DiffusionGemma rejects temperature and seed.
            url = f"http://{ip}:{port}/v1/completions"
            data = {
                "model": model,
                "prompt": message,
                "max_tokens": max(max_tokens, 32),
            }
            response = _post_local_json(url, headers, data)
            text_output = response["choices"][0]["text"]
        else:
            url = f"http://{ip}:{port}/v1/completions"
            data = {
                "model": model,
                "prompt": message,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": 10,
                "seed": seed,
            }
            response = _post_local_json(url, headers, data)
            text_output = response["choices"][0]["text"]
    else:
        model = "gpt-4.1-nano-2025-04-14" # Default model for OpenAI API
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        response: types.Completion = client.completions.create(
            model=model,
            prompt=message,
            max_tokens=5,
            temperature=1.0,
            logprobs=10,
            seed=seed,
        )
        
        text_output = response.choices[0].text
    
    return text_output

### QA ###
from openai import OpenAI
import os

def chat_perturb(
    message: str,
    seed: int,
    max_tokens: int = 512,
    model: str = "Qwen/Qwen2.5-14B",
    port: int = 8000,
    ip: str = "localhost",
):
    if is_llada_model(model):
        return generate_text(
            message,
            seed=seed,
            max_tokens=max_tokens,
            temperature=1.0,
            model_name=model,
        )

    if _is_diffusion_gemma(model):
        url = f"http://{ip}:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # DiffusionGemma rejects request-level temperature and seed.
        response_json = _post_local_json(url, headers, data)
        text_output = response_json["choices"][0]["message"]["content"] or ""
    else:
        client = OpenAI(
            api_key="dummy-key",
            base_url=f"http://{ip}:{port}/v1",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "/no_think You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
            temperature=1.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        text_output = completion.choices[0].message.content or ""

    return text_output

def chat_qa(
    message: str,
    label_keys,
    seed: int,
    model: str = "Qwen/Qwen2.5-14B",
    port: int = 8000,
    ip: str = "localhost",
):
    if is_llada_model(model):
        return score_labels(
            message,
            label_keys,
            seed=seed,
            temperature=1.0,
            model_name=model,
        )

    headers = {"Content-Type": "application/json"}
    label_logprobs = {}

    if _is_diffusion_gemma(model):
        url = f"http://{ip}:{port}/v1/chat/completions"
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the user's output-format instructions exactly.",
                },
                {"role": "user", "content": message},
            ],
            "max_tokens": 32,
            "logprobs": True,
            "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response_json = _post_local_json(url, headers, data)
        choice = response_json["choices"][0]
        text_output = choice["message"]["content"] or ""
        logprobs_list = (choice.get("logprobs") or {}).get("content", [])

        # Use alternatives at the position where DiffusionGemma generated its label.
        for token_info in logprobs_list:
            if token_info.get("token", "").strip() not in label_keys:
                continue
            for token_option in token_info.get("top_logprobs", []):
                stripped_token = token_option.get("token", "").strip()
                if stripped_token in label_keys:
                    label_logprobs[stripped_token] = token_option["logprob"]
            break
    else:
        url = f"http://{ip}:{port}/v1/completions"
        data = {
            "model": model,
            "prompt": message,
            "temperature": 1.0,
            "max_tokens": 10,
            "logprobs": 10,
            "seed": seed,
        }
        response_json = _post_local_json(url, headers, data)
        choice = response_json["choices"][0]
        text_output = choice["text"]
        logprobs_list = (choice.get("logprobs") or {}).get("top_logprobs", [])

        for token_probs in logprobs_list:
            for token_option, logprob in token_probs.items():
                stripped_token = token_option.strip()
                if stripped_token in label_keys:
                    if (
                        stripped_token not in label_logprobs
                        or logprob > label_logprobs[stripped_token]
                    ):
                        label_logprobs[stripped_token] = logprob

    # Normalize to probabilities
    exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}

    return text_output, normalized_probs
