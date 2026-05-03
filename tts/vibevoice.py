"""VibeVoice TTS wrapper for PaperNarrator.

Uses VibeVoice-Realtime-0.5B for streaming TTS inference.
"""

import os
import copy
from typing import Optional
import torch


class VibeVoiceTTS:
    """
    Wrapper for Microsoft VibeVoice-Realtime-0.5B TTS model.
    
    Generates long-form speech from text using streaming inference
    with pre-computed voice profiles (Carter, Davis, Emma).
    """
    
    # Hard limit: 0.5B model supports up to ~60 minutes at 150 wpm = ~9000 words.
    # We set a conservative ceiling of 10000 words.
    MAX_WORDS = 10000

    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: Optional[str] = None,
        speaker_name: str = "Carter",
        cfg_scale: float = 1.5,
        num_steps: int = 20
    ):
        """
        Initialize VibeVoice TTS.
        
        Args:
            model_name: Local model path or HuggingFace model ID
            device: 'cuda', 'mps', or 'cpu' (None for auto-detection)
            speaker_name: Voice profile name - one of: Carter, Davis, Emma
            cfg_scale: Classifier-Free Guidance scale (default: 1.5)
            num_steps: DDPM inference steps (default: 20, higher = better quality)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() 
                                else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.speaker_name = speaker_name
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps or int(os.getenv("VIBEVOICE_NUM_STEPS", 20))
        
        self.processor = None
        self.model = None
        self._loaded = False
        self._voice_sample = None
        
        print(f"VibeVoice TTS initialized.")
        print(f"Model: {model_name}, Device: {self.device}, Speaker: {speaker_name}")
        print("(Model will load on first generate_audio call)")
    
    def _load_model(self):
        """Lazy load model to save memory until needed."""
        if self._loaded:
            return
            
        print(f"Loading TTS model {self.model_name}...")
        
        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference
            )
            from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
            
            # Verify model has generate() before proceeding
            if not hasattr(VibeVoiceStreamingForConditionalGenerationInference, 'generate'):
                raise RuntimeError(
                    "VibeVoiceStreamingForConditionalGenerationInference is missing the "
                    "'generate' method. The installed vibevoice package may be incompatible."
                )
            
            # Load processor
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_name)
            
            # Determine dtype and attention implementation
            if self.device == "mps":
                load_dtype = torch.float32
                attn_impl = "sdpa"
            elif self.device == "cuda":
                load_dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
            
            print(f"Using dtype: {load_dtype}, attn_impl: {attn_impl}")
            
            # Load model
            try:
                if self.device == "mps":
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_name,
                        torch_dtype=load_dtype,
                        attn_implementation=attn_impl,
                        device_map=None,
                    )
                    self.model.to("mps")
                elif self.device == "cuda":
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_name,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                    )
                else:
                    self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_name,
                        torch_dtype=load_dtype,
                        device_map="cpu",
                        attn_implementation=attn_impl,
                    )
            except Exception as e:
                # Fallback to SDPA if flash_attention_2 fails
                if attn_impl == 'flash_attention_2':
                    print(f"Flash attention failed, falling back to SDPA: {e}")
                    if self.device == "mps":
                        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                            self.model_name,
                            torch_dtype=load_dtype,
                            attn_implementation='sdpa',
                            device_map=None,
                        )
                        self.model.to("mps")
                    elif self.device == "cuda":
                        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                            self.model_name,
                            torch_dtype=load_dtype,
                            device_map="cuda",
                            attn_implementation='sdpa',
                        )
                    else:
                        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                            self.model_name,
                            torch_dtype=load_dtype,
                            device_map="cpu",
                            attn_implementation='sdpa',
                        )
                else:
                    raise
            
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=self.num_steps)
            
            # Load voice sample
            voice_path = os.path.join(self.model_name, "voices", f"{self.speaker_name}.pt")
            if not os.path.exists(voice_path):
                raise FileNotFoundError(
                    f"Voice sample not found for '{self.speaker_name}'. "
                    f"Looked in: {voice_path}. "
                    f"Run download_vibevoice.py to download voices."
                )
            
            # Load voice preset
            # Note: weights_only=False is required because voice files contain
            # BaseModelOutputWithPast objects (not basic tensors).
            # Voice files are official Microsoft files from GitHub.
            self._voice_sample = torch.load(
                voice_path,
                map_location=self.device,
                weights_only=False
            )
            
            # Validate voice file structure
            if not isinstance(self._voice_sample, dict):
                raise ValueError(
                    f"Invalid voice file format for '{self.speaker_name}': "
                    f"expected dict, got {type(self._voice_sample)}"
                )
            
            expected_keys = {'lm', 'tts_lm', 'neg_lm', 'neg_tts_lm'}
            if not expected_keys.issubset(set(self._voice_sample.keys())):
                raise ValueError(
                    f"Invalid voice file for '{self.speaker_name}': "
                    f"missing keys {expected_keys - set(self._voice_sample.keys())}"
                )
            
            self._loaded = True
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model {self.model_name}: {e}") from e
    
    def generate_audio(
        self,
        text: str,
        output_path: str,
        word_count: Optional[int] = None
    ) -> str:
        """
        Generate audio from text using VibeVoice-Realtime-0.5B.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save WAV file
            word_count: Optional word count (will be calculated if not provided)
            
        Returns:
            Path to generated audio file
            
        Raises:
            ValueError: If word count exceeds MAX_WORDS limit
            RuntimeError: If generation fails
        """
        if not self._loaded:
            self._load_model()
        
        # Calculate word count if not provided
        if word_count is None:
            word_count = len(text.split())
        
        # Validate word count
        if word_count > self.MAX_WORDS:
            raise ValueError(
                f"Text exceeds {self.MAX_WORDS} words ({word_count} words). "
                f"VibeVoice-0.5B supports ~60 minutes of audio. "
                f"Please split into smaller chunks."
            )
        
        if word_count > 8500:
            print(f"Warning: Text has {word_count} words (>8500). This may take >50 minutes to generate.")
        
        print(f"Generating audio for {word_count} words...")
        
        try:
            # Prepare inputs using the streaming processor with cached voice
            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=self._voice_sample,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)
            
            # Generate audio using streaming inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
                all_prefilled_outputs=copy.deepcopy(self._voice_sample)
            )
            
            # Save output
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            self.processor.save_audio(
                outputs.speech_outputs[0],
                output_path=output_path
            )
            
            print(f"Audio saved to {output_path}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate audio: {e}") from e