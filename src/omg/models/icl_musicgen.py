"""ICL-enhanced MusicGen model for Text-to-Music generation."""

import torch
import torch.nn as nn
from audiocraft.models import MusicGen
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class ICLMusicGen(nn.Module):
    """MusicGen with In-Context Learning support.

    This model wraps MusicGen to support few-shot learning from
    (text, audio) example pairs.

    Optimized for RTX 5070 Ti (16GB VRAM).
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        use_lora: bool = True,
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.use_4bit = use_4bit

        # Load base model
        self.model = self._load_model()

        # Apply optimizations
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        if use_4bit:
            self._setup_quantization()

        if use_lora:
            self._setup_lora(lora_r, lora_alpha, lora_dropout)

        # Special tokens for ICL
        self.sep_token_id = None  # Will be set during tokenizer setup

    def _load_model(self) -> MusicGen:
        """Load the base MusicGen model."""
        # Extract model size from name
        if "small" in self.model_name:
            size = "small"
        elif "large" in self.model_name:
            size = "large"
        else:
            size = "medium"

        model = MusicGen.get_pretrained(size)
        return model

    def _setup_quantization(self):
        """Configure 4-bit quantization for memory efficiency."""
        # Note: MusicGen quantization requires custom handling
        # This is a placeholder for the actual implementation
        pass

    def _setup_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA adapters to the model."""
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply to the language model component
        if hasattr(self.model, 'lm'):
            self.model.lm = prepare_model_for_kbit_training(self.model.lm)
            self.model.lm = get_peft_model(self.model.lm, lora_config)

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory."""
        if hasattr(self.model, 'lm') and hasattr(self.model.lm, 'gradient_checkpointing_enable'):
            self.model.lm.gradient_checkpointing_enable()

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to discrete tokens using EnCodec.

        Args:
            audio: Waveform tensor of shape (batch, channels, samples)

        Returns:
            Encoded tokens of shape (batch, num_codebooks, seq_len)
        """
        with torch.no_grad():
            encoded = self.model.compression_model.encode(audio)
        return encoded[0]  # Return just the codes

    def prepare_icl_input(
        self,
        examples: list[tuple[str, torch.Tensor]],
        target_prompt: str,
    ) -> dict:
        """Prepare input sequence with ICL examples.

        Args:
            examples: List of (text_description, audio_tensor) pairs
            target_prompt: The target text prompt for generation

        Returns:
            Dictionary containing prepared inputs
        """
        # Encode all example audios
        encoded_examples = []
        for text, audio in examples:
            audio_tokens = self.encode_audio(audio)
            encoded_examples.append({
                "text": text,
                "audio_tokens": audio_tokens,
            })

        return {
            "examples": encoded_examples,
            "target_prompt": target_prompt,
        }

    def generate(
        self,
        prompt: str,
        icl_examples: list[tuple[str, torch.Tensor]] | None = None,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Generate music with optional ICL examples.

        Args:
            prompt: Text description for generation
            icl_examples: Optional list of (text, audio) example pairs
            duration: Duration in seconds
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated audio waveform
        """
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if icl_examples:
            # TODO: Implement ICL-aware generation
            # This requires modifying the generation loop to prepend
            # encoded examples to the context
            pass

        # Standard generation
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.model.generate([prompt])

        return output

    def forward(
        self,
        text_inputs: list[str],
        audio_inputs: torch.Tensor | None = None,
        icl_examples: list[tuple[str, torch.Tensor]] | None = None,
    ) -> dict:
        """Forward pass for training.

        Args:
            text_inputs: List of text prompts
            audio_inputs: Target audio for training
            icl_examples: Optional ICL examples

        Returns:
            Dictionary containing loss and other outputs
        """
        # Store inputs for future use
        _ = text_inputs, audio_inputs, icl_examples
        # This is a placeholder for the training forward pass
        # Full implementation requires custom loss computation
        raise NotImplementedError("Training forward pass not yet implemented")

    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        if self.use_lora and hasattr(self.model, 'lm'):
            trainable = sum(p.numel() for p in self.model.lm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.lm.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    @torch.no_grad()
    def estimate_memory_usage(self) -> dict:
        """Estimate GPU memory usage."""
        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())

        return {
            "parameters_mb": param_memory / 1024 / 1024,
            "estimated_training_mb": param_memory * 4 / 1024 / 1024,  # Rough estimate
        }
