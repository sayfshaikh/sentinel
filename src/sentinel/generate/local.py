from __future__ import annotations

from typing import Any, cast


class HFGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        context_limit: int = 4096,
        device: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.context_limit = context_limit
        self._temperature = temperature
        self._device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        loader = cast("Any", AutoModelForCausalLM.from_pretrained)
        self._model: Any = loader(model_name, torch_dtype="auto")
        self._model.to(self._device)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)
        print(inputs)

        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self._temperature > 0,
            temperature=self._temperature or None,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        completion = output[0][prompt_len:]
        return str(self._tokenizer.decode(completion, skip_special_tokens=True)).strip()
