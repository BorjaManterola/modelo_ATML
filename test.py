import json
import os
from typing import Dict, Union

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_from_adapter_config(adapter_dir: str) -> str:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get(
        "base_model_name_or_path",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )


def load_model_and_tokenizer(adapter_dir: str = "modelo_finetuned_local"):
    device = detect_device()
    # dtype conservador para CPU/MPS; float16 para CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_model_id = load_base_from_adapter_config(adapter_dir)

    # Cargar tokenizer desde el directorio del adaptador para conservar
    # special tokens y plantilla
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar modelo base y aplicar LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        torch_dtype=dtype,
    )

    model.eval()
    model.to(device)

    return model, tokenizer, device


def _as_model_inputs(
    tokens: Union[torch.Tensor, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if isinstance(tokens, torch.Tensor):
        return {"input_ids": tokens}
    return tokens


def build_inputs(prompt: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    # Intentar usar la plantilla de chat si existe en el tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            tok = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return _as_model_inputs(tok)
        except Exception:
            pass
    # Fallback a prompt plano
    return tokenizer(prompt, return_tensors="pt")


def generate_text(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    inputs = build_inputs(prompt, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)

    # Extraer solo los tokens generados (sin el prompt)
    gen_tokens = outputs[0, input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inferencia con LoRA finetuned (TinyLlama)"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Texto de entrada o instrucción del usuario",
    )
    parser.add_argument(
        "--adapter_dir",
        default="modelo_finetuned_local",
        help="Ruta al directorio del adaptador LoRA",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Cargar una sola vez
    model, tokenizer, device = load_model_and_tokenizer(args.adapter_dir)

    # Si se pasa --prompt, responder primero
    if args.prompt:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n=== Respuesta ===\n")
        print(output)

    # Modo interactivo
    print("\nEntrando en modo interactivo. Escribe 'exit' para salir.\n")
    while True:
        try:
            user_prompt = input("Usuario> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit", "salir"}:
            print("Saliendo...")
            break

        try:
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("\nModelo>\n")
            print(output)
            print()
        except Exception as e:
            print(f"Error durante la generación: {e}")


if __name__ == "__main__":
    main()
