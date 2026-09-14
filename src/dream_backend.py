"""Local HuggingFace backend for Dream dLLMs (no vLLM / SGLang)."""

from __future__ import annotations

import os
import re
import threading
from typing import Iterable

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

_LOCK = threading.Lock()
_MODEL = None
_TOKENIZER = None
_LOADED_NAME = None
_DREAM_GEN_CFG_CLS = None

# Inserted into ICL as the u label, then replaced by mask tokens.
# Class needle: <output>{PH}</output>. Reg needle: <output> {PH}  (keep " </output>").
U_PLACEHOLDER = "###VUD_U_PLACEHOLDER###"


def _patch_rope_default() -> None:
    """Dream's modeling file looks up ROPE_INIT_FUNCTIONS['default'], removed in transformers 5."""
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
        if rope_kwargs.get("dim") is not None:
            dim = int(rope_kwargs["dim"])
            base = float(rope_kwargs.get("base", 10000.0))
        else:
            base = float(getattr(config, "rope_theta", 10000.0))
            partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0) or 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
        freqs = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base ** freqs)
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_dream_generation_config(model_name: str) -> None:
    """Dream's validate() does not accept transformers 5's user_set_attributes kwarg."""
    global _DREAM_GEN_CFG_CLS
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "generation_utils.DreamGenerationConfig",
        model_name,
    )
    _DREAM_GEN_CFG_CLS = cls
    if getattr(cls.validate, "_vud_compat", False):
        return

    def validate(self, *args, **kwargs):
        return None

    validate._vud_compat = True
    cls.validate = validate


def _reset_dream_rope(model) -> None:
    """Recompute inv_freq on the model device; init-time RoPE can be garbage on transformers 5."""
    config = model.config
    device = next(model.parameters()).device
    base = float(getattr(config, "rope_theta", 1_000_000.0))
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim)
    freqs = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    inv_freq = 1.0 / (base ** freqs)
    ropes = [model.model.rotary_emb]
    for layer in model.model.layers:
        ropes.append(layer.self_attn.rotary_emb)
    for rope in ropes:
        rope.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        rope.original_inv_freq = rope.inv_freq
        rope.attention_scaling = 1.0


def _attach_generation_config(model, tokenizer, model_name: str, mask_id) -> None:
    """Hub generation_config.json plus Dream README sampling defaults."""
    cls = _DREAM_GEN_CFG_CLS
    if cls is None:
        if mask_id is not None:
            model.generation_config.mask_token_id = int(mask_id)
        return
    try:
        gen = cls.from_pretrained(model_name)
    except Exception:
        gen = cls.from_model_config(model.config)
    # Hub json omits mask_token_id; config.json has 151666.
    if getattr(gen, "mask_token_id", None) is None and mask_id is not None:
        gen.mask_token_id = int(mask_id)
    if getattr(gen, "pad_token_id", None) is None:
        gen.pad_token_id = tokenizer.pad_token_id
    if getattr(gen, "eos_token_id", None) is None:
        gen.eos_token_id = tokenizer.eos_token_id
    if getattr(gen, "bos_token_id", None) is None:
        gen.bos_token_id = tokenizer.bos_token_id
    # Hub class defaults are greedy (temp 0). VUD generate uses --model_temperature (1).
    gen.eps = getattr(gen, "eps", None) or 1e-3
    gen.steps = 512
    gen.alg = "entropy"
    gen.alg_temp = 0.0
    gen.temperature = 1.0
    gen.top_p = 0.95
    gen.output_history = True
    gen.return_dict_in_generate = True
    gen.max_new_tokens = 512
    model.generation_config = gen


def _target_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if "CUDA_VISIBLE_DEVICES" in os.environ or torch.cuda.device_count() == 1:
        return torch.device("cuda")
    return torch.device("cuda:1")


def is_dream_model(model: str) -> bool:
    name = model.lower().replace("_", "-")
    return "dream-v0" in name or "dream-org/dream" in name


def _is_instruct(model: str) -> bool:
    return "instruct" in model.lower()


def _load(model_name: str):
    global _MODEL, _TOKENIZER, _LOADED_NAME
    with _LOCK:
        if _MODEL is not None and _LOADED_NAME == model_name:
            return _MODEL, _TOKENIZER

        _patch_rope_default()
        _patch_dream_generation_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        model = model.to(_target_device()).eval()
        _reset_dream_rope(model)
        mask_id = tokenizer.mask_token_id or getattr(model.config, "mask_token_id", None)
        if mask_id is not None:
            model.config.mask_token_id = int(mask_id)
        _attach_generation_config(model, tokenizer, model_name, mask_id)
        _MODEL = model
        _TOKENIZER = tokenizer
        _LOADED_NAME = model_name
        return _MODEL, _TOKENIZER


def _encode_prompt(message: str, tokenizer, model_name: str, device: torch.device):
    if _is_instruct(model_name):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    else:
        inputs = tokenizer(message, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    return input_ids, attention_mask


def _mask_token_id(model, tokenizer) -> int:
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(model.generation_config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        raise RuntimeError("Dream tokenizer has no mask token id.")
    return int(mask_id)


def _label_token_id(tokenizer, label: str) -> int:
    ids = tokenizer.encode(str(label), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer produced no ids for label {label!r}")
    return ids[0]


def _label_score_prefix(message, tokenizer, model_name, device):
    """Shared prefix for forward label scoring and the QA joint canvas."""
    message = (
        message.replace("{{label_prediction}}</output>", "")
        .replace("{{value_prediction}}</output>", "")
    )
    input_ids, _ = _encode_prompt(message, tokenizer, model_name, device)
    if _is_instruct(model_name):
        prefill_ids = tokenizer.encode("<output>", add_special_tokens=False)
        if prefill_ids:
            extra = torch.tensor([prefill_ids], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, extra], dim=1)
    return input_ids


@torch.no_grad()
def score_labels(
    message: str,
    label_keys: Iterable,
    seed: int = 0,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
):
    """Score 0/1 via Dream's eval loglikelihood forward (shifted logits under bf16 autocast)."""
    del seed
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    # QA templates end with <output>{{label_prediction}}</output>; score the token after <output>.
    input_ids = _label_score_prefix(message, tokenizer, model_name, device)

    mask_id = _mask_token_id(model, tokenizer)
    x = torch.cat(
        [input_ids, torch.full((1, 1), mask_id, device=device, dtype=input_ids.dtype)],
        dim=1,
    )

    # Same calling convention as DreamLM/Dream eval.get_logits / _eval_target_nll_*.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    token_logits = logits[0, input_ids.shape[1]].float()
    if not torch.isfinite(token_logits).all():
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x).logits.float()
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        token_logits = logits[0, input_ids.shape[1]]
    if temperature and temperature > 0:
        token_logits = token_logits / temperature

    keys = [str(k) for k in label_keys]
    selected = torch.stack([token_logits[_label_token_id(tokenizer, k)] for k in keys])
    if not torch.isfinite(selected).all():
        selected = torch.nan_to_num(selected, nan=-1e9)
    probs = torch.softmax(selected, dim=0)
    normalized = {k: float(probs[i]) for i, k in enumerate(keys)}
    pred = max(normalized, key=normalized.get)
    return f"<output>{pred}</output>", normalized


@torch.no_grad()
def generate_text(
    message: str,
    seed: int = 0,
    max_tokens: int = 512,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    input_ids, attention_mask = _encode_prompt(message, tokenizer, model_name, device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # Official Dream README generate kwargs. Must be a DreamGenerationConfig:
    # transformers 5's from_model_config returns a vanilla GenerationConfig (no eps).
    gen_cls = _DREAM_GEN_CFG_CLS
    if gen_cls is None:
        raise RuntimeError("DreamGenerationConfig was not loaded.")
    n_new = max(int(max_tokens), 1)
    sample_temp = 1.0 if temperature is None else float(temperature)
    gen_cfg = gen_cls(
        max_new_tokens=n_new,
        output_history=False,
        return_dict_in_generate=True,
        steps=n_new,
        temperature=sample_temp,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        eps=1e-3,
        mask_token_id=_mask_token_id(model, tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_len = input_ids.shape[1]
    text = tokenizer.decode(sequences[0][prompt_len:].tolist())
    eos = tokenizer.eos_token
    if eos:
        text = text.split(eos)[0]
    return text


def _shifted_logits(model, x: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    if not torch.isfinite(logits).all():
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x).logits.float()
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits
    return logits.float()


def _label_probs_from_logits(token_logits, tokenizer, label_keys, temperature: float):
    keys = [str(k) for k in label_keys]
    selected = torch.stack([token_logits[_label_token_id(tokenizer, k)] for k in keys])
    if not torch.isfinite(selected).all():
        selected = torch.nan_to_num(selected, nan=-1e9)
    if temperature and temperature > 0:
        selected = selected / temperature
    probs = torch.softmax(selected, dim=0)
    return keys, probs


def _sample_label_from_logits(token_logits, tokenizer, label_keys, temperature: float) -> str:
    keys, probs = _label_probs_from_logits(token_logits, tokenizer, label_keys, temperature)
    idx = int(torch.multinomial(probs, 1).item())
    return keys[idx]


def _label_entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum())


def _probs_dict(keys, probs) -> dict[str, float]:
    return {k: float(probs[i]) for i, k in enumerate(keys)}


def _dream_gen_cfg(gen_cls, tokenizer, mask_id, *, max_new_tokens: int, steps: int, temperature: float):
    return gen_cls(
        max_new_tokens=max(int(max_new_tokens), 1),
        output_history=False,
        return_dict_in_generate=True,
        steps=max(int(steps), 1),
        temperature=1.0 if temperature is None else float(temperature),
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
        eps=1e-3,
        mask_token_id=mask_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )


def _vanilla_aligned_canvas(
    icl: str,
    x_note: str,
    tokenizer,
    mask_id: int,
    n_u_masks: int,
    n_y_masks: int,
    device,
    model_name: str,
    placeholder: str = U_PLACEHOLDER,
    spaced_placeholder: bool = False,
):
    """Vanilla ICL with u replaced by masks, plus y masks. No suffix after y.

    `icl` must contain the placeholder as the u label (z shuffled in with D).
    Class: split on PH so left ends at `<output>`. Reg: split on ` {PH}` so the
    mask span carries the leading-space token and rest keeps ` </output>`
    (tokenizes as Ġ</, not </). Prefix uses tokenizer() like `_encode_prompt`.
    """
    if _is_instruct(model_name):
        raise NotImplementedError(
            "Vanilla-aligned canvas is implemented for Dream Base toy prompts, not Instruct."
        )
    needle = f" {placeholder}" if spaced_placeholder else placeholder
    if needle not in icl:
        raise ValueError(
            f"placeholder needle {needle!r} not found in ICL "
            f"(placeholder={placeholder!r})"
        )
    left, rest = icl.split(needle, 1)
    mid = rest + f"\n {x_note} <output>"
    pre = tokenizer(left, return_tensors="pt").input_ids[0].tolist()
    mid_ids = tokenizer.encode(mid, add_special_tokens=False)
    ids = pre + [mask_id] * n_u_masks + mid_ids + [mask_id] * n_y_masks
    u_pos = len(pre)
    y_pos = len(pre) + n_u_masks + len(mid_ids)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    return x, u_pos, y_pos


def infer_number_n_masks(labels: Iterable, model_name: str) -> int:
    """Mask-span length from D label formatting (` {label}` after `<output>`)."""
    _, tokenizer = _load(model_name)
    n = 1
    for lab in labels:
        ids = tokenizer.encode(f" {lab}", add_special_tokens=False)
        n = max(n, len(ids))
    return n


def _qa_joint_canvas(message, tokenizer, mask_id, device, model_name):
    """Mask u inside the full QA user prompt and y in the assistant response.

    Uses exactly the forward scorer's chat template and assistant prefill.
    Both slots are on one bidirectional canvas, so either can be filled first.
    """
    if message.count(U_PLACEHOLDER) != 1:
        raise ValueError("QA joint prompt must contain exactly one U_PLACEHOLDER.")
    mask_text = tokenizer.convert_ids_to_tokens(mask_id)
    message = message.replace(U_PLACEHOLDER, mask_text)
    prefix = _label_score_prefix(message, tokenizer, model_name, device)
    positions = (prefix[0] == mask_id).nonzero(as_tuple=False).flatten()
    if len(positions) != 1:
        raise ValueError("QA joint prompt must encode exactly one auxiliary mask.")
    y_pos = prefix.shape[1]
    canvas = torch.cat([
        prefix,
        torch.full((1, 1), mask_id, device=device, dtype=prefix.dtype),
    ], dim=1)
    return canvas, int(positions[0].item()), y_pos


@torch.no_grad()
def sample_two_label_slots(
    icl: str,
    x_note: str,
    label_keys: Iterable,
    seed: int = 0,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
    message: str | None = None,
) -> tuple[str, str]:
    """Jointly denoise (u, y); optionally use a full QA prompt (including Instruct)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    label_keys = list(label_keys)
    label_ids = [tokenizer.encode(str(k), add_special_tokens=False) for k in label_keys]
    if not label_ids or any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Joint label sampling requires nonempty, single-token labels.")
    if len({ids[0] for ids in label_ids}) != len(label_ids):
        raise ValueError("Joint labels must have distinct token IDs.")
    if message is None:
        x, u_pos, y_pos = _vanilla_aligned_canvas(
            icl, x_note, tokenizer, mask_id, 1, 1, device, model_name
        )
    else:
        x, u_pos, y_pos = _qa_joint_canvas(
            message, tokenizer, mask_id, device, model_name
        )

    remaining = {"u": u_pos, "y": y_pos}
    filled = {}
    while remaining:
        logits = _shifted_logits(model, x)
        best_name = None
        best_ent = None
        dists = {}
        for name, pos in remaining.items():
            keys, probs = _label_probs_from_logits(
                logits[0, pos], tokenizer, label_keys, temperature
            )
            dists[name] = (keys, probs)
            ent = _label_entropy(probs)
            # Dream/LLaDA entropy transfer: lowest entropy (most confident) first.
            if best_ent is None or ent < best_ent:
                best_name, best_ent = name, ent
        keys, probs = dists[best_name]
        token = keys[int(torch.multinomial(probs, 1).item())]
        filled[best_name] = token
        x[0, remaining[best_name]] = _label_token_id(tokenizer, token)
        del remaining[best_name]
    return filled["u"], filled["y"]


@torch.no_grad()
def score_two_label_marginals(
    icl: str,
    x_note: str,
    label_keys: Iterable,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> tuple[dict[str, float], dict[str, float]]:
    """Both slots masked: softmax at u_pos and y_pos (one forward)."""
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    x, u_pos, y_pos = _vanilla_aligned_canvas(
        icl, x_note, tokenizer, mask_id, 1, 1, device, model_name
    )
    logits = _shifted_logits(model, x)
    u_keys, u_probs = _label_probs_from_logits(logits[0, u_pos], tokenizer, label_keys, temperature)
    y_keys, y_probs = _label_probs_from_logits(logits[0, y_pos], tokenizer, label_keys, temperature)
    return _probs_dict(u_keys, u_probs), _probs_dict(y_keys, y_probs)


@torch.no_grad()
def score_two_label_joint(
    icl: str,
    x_note: str,
    label_keys: Iterable,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float], str]:
    """Exact joint law of the constrained two-slot entropy denoiser.

    The first slot is selected using the same minimum-entropy rule as
    :func:`sample_two_label_slots`.  The selected token is then enumerated and
    the other slot is rescored, giving the complete joint distribution without
    Monte Carlo error.
    """
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    canvas, u_pos, y_pos = _vanilla_aligned_canvas(
        icl, x_note, tokenizer, mask_id, 1, 1, device, model_name
    )

    logits = _shifted_logits(model, canvas)
    u_keys, u_probs = _label_probs_from_logits(
        logits[0, u_pos], tokenizer, label_keys, temperature
    )
    y_keys, y_probs = _label_probs_from_logits(
        logits[0, y_pos], tokenizer, label_keys, temperature
    )
    initial_u = _probs_dict(u_keys, u_probs)
    initial_y = _probs_dict(y_keys, y_probs)
    joint: dict[tuple[str, str], float] = {}

    if _label_entropy(u_probs) <= _label_entropy(y_probs):
        first_slot = "u"
        for u_idx, u in enumerate(u_keys):
            conditioned = canvas.clone()
            conditioned[0, u_pos] = _label_token_id(tokenizer, u)
            conditional_logits = _shifted_logits(model, conditioned)
            conditional_y_keys, conditional_y_probs = _label_probs_from_logits(
                conditional_logits[0, y_pos], tokenizer, y_keys, temperature
            )
            for y_idx, y in enumerate(conditional_y_keys):
                joint[(u, y)] = float(u_probs[u_idx] * conditional_y_probs[y_idx])
    else:
        first_slot = "y"
        for y_idx, y in enumerate(y_keys):
            conditioned = canvas.clone()
            conditioned[0, y_pos] = _label_token_id(tokenizer, y)
            conditional_logits = _shifted_logits(model, conditioned)
            conditional_u_keys, conditional_u_probs = _label_probs_from_logits(
                conditional_logits[0, u_pos], tokenizer, u_keys, temperature
            )
            for u_idx, u in enumerate(conditional_u_keys):
                joint[(u, y)] = float(y_probs[y_idx] * conditional_u_probs[u_idx])

    total = sum(joint.values())
    if not total > 0:
        raise ValueError("Joint label distribution has zero mass.")
    joint = {pair: probability / total for pair, probability in joint.items()}
    return joint, initial_u, initial_y, first_slot


@torch.no_grad()
def score_y_given_u_label(
    icl: str,
    x_note: str,
    u: str,
    label_keys: Iterable,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> dict[str, float]:
    """pP(y | x, u, z, D): u filled on the two-mask canvas, softmax at the y mask."""
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    x, u_pos, y_pos = _vanilla_aligned_canvas(
        icl, x_note, tokenizer, mask_id, 1, 1, device, model_name
    )
    x[0, u_pos] = _label_token_id(tokenizer, u)
    logits = _shifted_logits(model, x)
    keys, probs = _label_probs_from_logits(logits[0, y_pos], tokenizer, label_keys, temperature)
    return _probs_dict(keys, probs)


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


def _parse_float_span(text: str) -> float:
    match = _FLOAT_RE.search(text.replace(",", ""))
    if not match:
        raise ValueError(f"no number in {text!r}")
    return float(match.group(0))


def _top_p_logits(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(remove).scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


@torch.no_grad()
def _denoise_existing_masks(
    model,
    input_ids: torch.Tensor,
    mask_id: int,
    *,
    steps: int,
    temperature: float,
    top_p: float = 0.95,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Dream entropy denoising restricted to masks already in ``input_ids``.

    Dream's public generation method always appends ``max_new_tokens`` masks.
    Parallel VUD has already placed both target spans on the canvas, so calling
    that method would introduce an unintended third target.  This is the same
    entropy-transfer loop, but it never changes the canvas length.
    """
    x = input_ids.clone()
    steps = max(int(steps), 1)
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

    for step in range(steps):
        mask_positions = torch.nonzero(x[0] == mask_id, as_tuple=False).flatten()
        if mask_positions.numel() == 0:
            break

        logits = _shifted_logits(model, x)[0, mask_positions]
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        if temperature is not None and temperature > 0:
            logits = logits / float(temperature)
        logits = _top_p_logits(logits, top_p)
        probs = torch.softmax(logits, dim=-1)

        if temperature is not None and temperature > 0:
            proposals = torch.multinomial(probs, 1).squeeze(-1)
        else:
            proposals = probs.argmax(dim=-1)
        confidence = torch.sum(probs * torch.log(probs.clamp_min(1e-10)), dim=-1)

        if step < steps - 1:
            transfer_fraction = 1 - timesteps[step + 1] / timesteps[step]
            n_transfer = int(mask_positions.numel() * transfer_fraction)
        else:
            n_transfer = int(mask_positions.numel())
        if n_transfer <= 0:
            continue

        chosen = torch.topk(confidence, min(n_transfer, confidence.numel())).indices
        x[0, mask_positions[chosen]] = proposals[chosen]

    if torch.any(x == mask_id):
        raise RuntimeError("Entropy denoising ended with unfilled target masks.")
    return x


@torch.no_grad()
def sample_two_number_slots(
    icl: str,
    x_note: str,
    seed: int = 0,
    n_masks: int = 8,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> tuple[float, float, list[int]]:
    """Jointly fill two numeric <output> slots (both masked, one diffusion run)."""
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_masks = max(int(n_masks), 1)
    input_ids, u_pos, y_pos = _vanilla_aligned_canvas(
        icl,
        x_note,
        tokenizer,
        mask_id,
        n_masks,
        n_masks,
        device,
        model_name,
        spaced_placeholder=True,
    )
    sequences = _denoise_existing_masks(
        model,
        input_ids,
        mask_id,
        steps=2 * n_masks,
        temperature=temperature,
    )
    seq = sequences[0]
    u_ids = seq[u_pos : u_pos + n_masks].tolist()
    y_text = tokenizer.decode(seq[y_pos : y_pos + n_masks].tolist())
    u_text = tokenizer.decode(u_ids)
    print(f"number span raw u={u_text!r} y={y_text!r}", flush=True)
    return _parse_float_span(u_text), _parse_float_span(y_text), u_ids


@torch.no_grad()
def sample_y_given_u_ids(
    icl: str,
    x_note: str,
    u_ids: list[int],
    seed: int = 0,
    n_masks: int = 8,
    temperature: float = 1.0,
    model_name: str = "Dream-org/Dream-v0-Base-7B",
) -> float:
    """Sample y from pP(y | x, u, z, D): freeze u-span tokens, denoise the y span only."""
    model, tokenizer = _load(model_name)
    device = next(model.parameters()).device
    mask_id = _mask_token_id(model, tokenizer)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_masks = max(int(n_masks), 1)
    input_ids, u_pos, y_pos = _vanilla_aligned_canvas(
        icl,
        x_note,
        tokenizer,
        mask_id,
        n_masks,
        n_masks,
        device,
        model_name,
        spaced_placeholder=True,
    )
    if len(u_ids) != n_masks:
        raise ValueError(f"u_ids length {len(u_ids)} != n_masks {n_masks}")
    input_ids[0, u_pos : u_pos + n_masks] = torch.tensor(
        u_ids, device=device, dtype=input_ids.dtype
    )
    sequences = _denoise_existing_masks(
        model,
        input_ids,
        mask_id,
        steps=n_masks,
        temperature=temperature,
    )
    seq = sequences[0]
    y_text = tokenizer.decode(seq[y_pos : y_pos + n_masks].tolist())
    return _parse_float_span(y_text)
