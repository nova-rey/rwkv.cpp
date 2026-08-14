import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from miditok import MMM, TokSequence
from transformers import LogitsProcessorList, GenerationConfig
from . import rwkv_cpp_shared_library, rwkv_cpp_model
from logits_processor import canonical_structural_token_replacements, decoded_token_names


class InvalidProbabilitySupportError(RuntimeError):
    """Raised when filtering leaves no numerically valid sampling support."""


def _tensor_stats(values: torch.Tensor) -> dict:
    array = values.detach().cpu()
    finite = torch.isfinite(array)
    result = {
        "shape": list(array.shape),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(array).sum().item()),
        "posinf_count": int(torch.isposinf(array).sum().item()),
        "neginf_count": int(torch.isneginf(array).sum().item()),
    }
    if finite.any():
        finite_values = array[finite]
        result["min_finite"] = float(finite_values.min().item())
        result["max_finite"] = float(finite_values.max().item())
    else:
        result["min_finite"] = None
        result["max_finite"] = None
    return result


class _SamplingTrace:
    def __init__(self, tokenizer):
        self.enabled = os.environ.get("MIDI_RWKV_SAMPLING_TRACE", "") == "1"
        self.tokenizer = tokenizer
        self.records = []
        self.path = None
        if self.enabled:
            trace_dir = os.environ.get("MIDI_RWKV_SAMPLING_TRACE_DIR", "trace")
            os.makedirs(trace_dir, exist_ok=True)
            self.path = os.path.join(trace_dir, "sampling_trace.jsonl")
            open(self.path, "w", encoding="utf-8").close()

    def record(self, record: dict):
        if not self.enabled:
            return
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self):
        if not self.enabled or self.path is None:
            return
        with open(self.path, "w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        summary_path = os.path.join(os.path.dirname(self.path), "sampling_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({"steps": len(self.records), "trace_path": self.path}, handle, indent=2)


def _top_tokens(scores: torch.Tensor, tokenizer, count: int = 20) -> list[dict]:
    values, indices = torch.topk(scores[0], min(count, scores.shape[-1]))
    result = []
    for value, index in zip(values.tolist(), indices.tolist()):
        sequence = TokSequence(ids=[int(index)], are_ids_encoded=True)
        tokenizer.decode_token_ids(sequence)
        result.append({"id": int(index), "logit": float(value), "tokens": list(sequence.tokens or [])})
    return result


def _semantic_tail(tokenizer, sequence: list[int], count: int = 20) -> list[dict]:
    """Decode the most recent generated IDs for actionable failure reports."""
    result = []
    for token_id in sequence[-count:]:
        decoded = TokSequence(ids=[int(token_id)], are_ids_encoded=True)
        tokenizer.decode_token_ids(decoded)
        result.append({"id": int(token_id), "tokens": list(decoded.tokens or [])})
    return result


def semantic_bar_count(tokenizer, sequence: list[int]) -> int:
    """Count bar boundaries by decoded MMM semantics, not one BPE id.

    The bundled tokenizer contains standalone, split, and compound encodings
    for Bar_None.  Counting only ``vocab["Bar_None"]`` under-counts valid
    prompt and generated sequences.
    """
    return sum("Bar_None" in decoded_token_names(tokenizer, token_id) for token_id in sequence)


def _validate_probability_support(
    scores: torch.Tensor,
    step: int,
    bar: int,
    trace: _SamplingTrace,
    tokenizer=None,
    sequence: list[int] | None = None,
    constraint_rules: dict[str, set[int]] | None = None,
) -> torch.Tensor:
    if not torch.isfinite(scores).any():
        details = {
            "last_semantic_tokens": _semantic_tail(tokenizer, sequence or []) if tokenizer else [],
            "active_constraint_rules": {
                rule: sorted(int(token_id) for token_id in token_ids)
                for rule, token_ids in (constraint_rules or {}).items()
            },
        }
        raise InvalidProbabilitySupportError(
            f"No finite logit support at generation step {step}, bar {bar}; "
            f"details={json.dumps(details, sort_keys=True)}"
        )
    probs = F.softmax(scores, dim=-1)
    probability_sum = probs.sum()
    if (
        not torch.isfinite(probs).all()
        or not torch.isfinite(probability_sum)
        or probability_sum <= 0
        or not torch.any(probs > 0)
    ):
        details = {
            "last_semantic_tokens": _semantic_tail(tokenizer, sequence or []) if tokenizer else [],
            "active_constraint_rules": {
                rule: sorted(int(token_id) for token_id in token_ids)
                for rule, token_ids in (constraint_rules or {}).items()
            },
        }
        raise InvalidProbabilitySupportError(
            f"Invalid probability support at generation step {step}, bar {bar}: "
            f"finite_logits={torch.isfinite(scores).sum().item()} "
            f"probability_sum={probability_sum.item()} "
            f"details={json.dumps(details, sort_keys=True)}"
        )
    trace.record({
        "stage": "final_softmax",
        "step_index": step,
        "bar_index": bar,
        "stats": _tensor_stats(probs),
        "probability_sum": float(probability_sum.item()),
        "nonzero_probability_count": int((probs > 0).sum().item()),
    })
    return probs


def apply_repetition_penalty(scores: torch.Tensor, previous_token_ids: set[int], penalty: float) -> torch.Tensor:
    """Apply the standard sign-aware repetition penalty in place."""
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0:
        return scores
    for token_id in previous_token_ids:
        if token_id >= scores.size(-1):
            continue
        if scores[0, token_id] > 0:
            scores[0, token_id] /= penalty
        else:
            scores[0, token_id] *= penalty
    return scores


class CppModelConfig:
    """Configuration class for the C++ model wrapper."""
    model_type: str = "cpp_model"
    
    def __init__(
        self,
        model_path: str = "",
        state_path: str = "",
        gpu_layer_count: int | None = None,
        thread_count: int | None = None,
        shared_library_path: str | None = None,
        **kwargs
    ):
        self.model_path = model_path
        self.state_path = state_path
        self.shared_library_path = shared_library_path or os.environ.get("MIDI_RWKV_SHARED_LIBRARY")
        if gpu_layer_count is None:
            raw_gpu_layers = os.environ.get("MIDI_RWKV_GPU_LAYERS", "0")
            try:
                gpu_layer_count = int(raw_gpu_layers)
            except ValueError as exc:
                raise ValueError("MIDI_RWKV_GPU_LAYERS must be an integer") from exc
        if gpu_layer_count < 0:
            raise ValueError("gpu_layer_count must be non-negative")
        self.gpu_layer_count = gpu_layer_count
        if thread_count is None:
            raw_threads = os.environ.get("MIDI_RWKV_THREADS")
            thread_count = int(raw_threads) if raw_threads else None
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread_count must be positive")
        self.thread_count = thread_count


class CustomGenerator:
    def __init__(self, config: CppModelConfig, tokenizer: MMM):
        # Load the C++ model
        if config.shared_library_path:
            self.library = rwkv_cpp_shared_library.RWKVSharedLibrary(config.shared_library_path)
        else:
            self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        self.model = rwkv_cpp_model.RWKVModel(
            self.library, 
            config.model_path, 
            thread_count=(config.thread_count if config.thread_count is not None else max(1, os.cpu_count() // 2)),
            gpu_layer_count=config.gpu_layer_count
        )
        self.tokenizer = tokenizer    
        self._sampling_trace = _SamplingTrace(tokenizer)
        self.current_state = None
        self.state = self.initialize_with_tuned_state(config.state_path)

        self.tokens_ending_bar_none = []
        self.tokens_beginning_timesig = []
        self.tokens_have_bar_none_and_timesig = []
        for i in range(tokenizer.vocab_size):
            t = TokSequence(ids=[i], are_ids_encoded=True)
            tokenizer.decode_token_ids(t)
            if len(t.tokens) == 0:
                continue
            if t.tokens[-1] == "Bar_None":
                self.tokens_ending_bar_none.append(i)
            if "TimeSig" in t.tokens[0]:
                self.tokens_beginning_timesig.append(i)
            if "Bar_None" in t.tokens and any("TimeSig" in x for x in t.tokens):
                self.tokens_have_bar_none_and_timesig.append(i)
        self.structural_token_replacements = canonical_structural_token_replacements(tokenizer)
        self._ac_trace_enabled = os.environ.get("MIDI_RWKV_AC_TRACE", "") == "1"
        self._ac_trace_path = None
        self._ac_trace_records = []
        if self._ac_trace_enabled:
            trace_dir = os.environ.get("MIDI_RWKV_AC_TRACE_DIR", "trace")
            os.makedirs(trace_dir, exist_ok=True)
            self._ac_trace_path = os.path.join(trace_dir, "ac_injection_trace.jsonl")
            open(self._ac_trace_path, "w", encoding="utf-8").close()

    def initialize_with_tuned_state(self, state_path):
        """
        Initialize the model with pre-tuned state tensors from a state dictionary.
        
        Parameters:
            model: The RWKV model instance
            state_dict: The state dictionary containing the tuned state tensors
        
        Returns:
            initial_state: Combined NumPy array ready to be used with model.eval
        """
        if not state_path:
            return None
        import numpy as np
        import torch
        
        n_layer = self.model.n_layer
        n_embd = self.model.n_embed
        
        # Initialize components for each layer
        all_states = []
        
        # Load the state dictionary using torch.load and convert to numpy
        state_dict = torch.load(state_path, map_location="cpu")
        
        for layer_idx in range(n_layer):
            # 1. Create zero array for attention token shift
            att_token_shift = np.zeros((1, n_embd), dtype=np.float32)
            
            # 2. Create zero array for FFN token shift
            ffn_token_shift = np.zeros((1, n_embd), dtype=np.float32)
            
            # 3. Get the pre-tuned WKV state for this layer
            state_key = f"blocks.{layer_idx}.att.time_state"
            if state_key in state_dict:
                wkv_state = state_dict[state_key].numpy()  # Convert to numpy
                # Extract dimensions and reshape
                head_size = wkv_state.shape[1]
                wkv_state_reshaped = wkv_state.reshape(head_size, n_embd)
            else:
                # If key not found, create a default zero array
                print(f"Warning: {state_key} not found in state dict")
                wkv_state_reshaped = np.zeros((n_embd, n_embd), dtype=np.float32)
            
            # Concatenate the three components for this layer
            layer_state = np.concatenate([
                att_token_shift.flatten(),
                ffn_token_shift.flatten(),
                wkv_state_reshaped.flatten()
            ])
            
            all_states.append(layer_state)
        
        # Concatenate all layer states
        initial_state = np.concatenate(all_states)
        return initial_state
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig = None,
        logits_processor: LogitsProcessorList = None,
        attribute_controls: list = None,
    ) -> torch.LongTensor:
        self._sampling_trace = _SamplingTrace(self.tokenizer)
        self._ac_trace_records = []
        batch_size = input_ids.shape[0]
        
        if batch_size > 1:
            raise ValueError("Batched generation is not yet supported")
        
        # Process initial input sequence
        input_sequence = input_ids[0].cpu().tolist()
        current_sequence = input_sequence.copy()
        
        # Initialize state with the entire input sequence
        state = self.state.copy() if self.state is not None else None
        # attribnute controls are preinjected for bar infilling
        logits, current_state = self.model.eval_sequence_in_chunks(
            input_sequence, state, state, None, use_numpy=True
        )
        
        # Track previous tokens for repetition penalty
        prev_tokens_set = set(input_sequence)
        
        # Convert logits to tensor on the device
        logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
        
        # Keep track of tokens that were actually generated (not injected)
        tokens_generated = 0
        did_last_token_end_in_bar_none = False
        ac_idx = 1
        pending_ac_event = None

        while tokens_generated < generation_config.max_new_tokens:
            # Convert logits to next_token_logits format (batch_size, vocab_size)
            next_token_logits = logits_tensor.clone()
            bar_index = semantic_bar_count(self.tokenizer, current_sequence)
            trace = {
                "step_index": tokens_generated,
                "bar_index": int(bar_index),
                "raw_logits": _tensor_stats(next_token_logits),
                "top_20_token_ids_before_filtering": _top_tokens(next_token_logits, self.tokenizer),
            }
            self._sampling_trace.record(trace)
            if not torch.isfinite(next_token_logits).any():
                raise InvalidProbabilitySupportError(
                    f"No finite raw logits at generation step {tokens_generated}, bar {bar_index}"
                )

            next_token_scores = logits_processor(current_sequence, next_token_logits) if logits_processor else next_token_logits
            self._sampling_trace.record({"stage": "after_logits_processor", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores)})
            
            # Apply temperature scaling
            if generation_config.temperature > 0 and generation_config.temperature != 1.0:
                next_token_scores = next_token_scores / generation_config.temperature
            self._sampling_trace.record({"stage": "after_temperature", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores)})
            
            # Apply repetition penalty
            if generation_config.repetition_penalty != 1.0:
                apply_repetition_penalty(
                    next_token_scores,
                    prev_tokens_set,
                    generation_config.repetition_penalty,
                )
            self._sampling_trace.record({"stage": "after_repetition_penalty", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores)})
            
            # Apply epsilon cutoff
            if generation_config.epsilon_cutoff > 0:
                # Create a mask for tokens below the probability threshold
                probs = F.softmax(next_token_scores, dim=-1)
                next_token_scores[probs < generation_config.epsilon_cutoff] = -float('inf')
            self._sampling_trace.record({"stage": "after_epsilon_cutoff", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores), "surviving_support_count": int(torch.isfinite(next_token_scores).sum().item())})
            
            if generation_config.do_sample:
                # Apply top-k filtering
                if 0 < generation_config.top_k < next_token_scores.size(-1):
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_scores, generation_config.top_k, dim=-1, largest=True, sorted=True
                    )
                    
                    # Create a new tensor with -inf everywhere
                    filtered_logits = torch.full_like(next_token_scores, -float('inf'))
                    
                    # Scatter the top-k logits back to the original tensor
                    filtered_logits[0, top_k_indices[0]] = top_k_logits[0]
                    
                    next_token_scores = filtered_logits
                self._sampling_trace.record({"stage": "after_top_k", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores), "surviving_support_count": int(torch.isfinite(next_token_scores).sum().item())})
                
                # Apply top-p (nucleus) filtering
                if generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_scores, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set removed indices to -inf
                    indices_to_remove = sorted_indices[0][sorted_indices_to_remove[0]]
                    next_token_scores[0, indices_to_remove] = -float('inf')
                self._sampling_trace.record({"stage": "after_top_p", "step_index": tokens_generated, "bar_index": int(bar_index), "stats": _tensor_stats(next_token_scores), "surviving_support_count": int(torch.isfinite(next_token_scores).sum().item())})
                
                # Convert logits to probabilities and sample
                probs = _validate_probability_support(
                    next_token_scores,
                    tokens_generated,
                    int(bar_index),
                    self._sampling_trace,
                    tokenizer=self.tokenizer,
                    sequence=current_sequence,
                    constraint_rules=next(
                        (
                            getattr(processor, "constraint_token_ids")
                            for processor in (logits_processor or [])
                            if hasattr(processor, "constraint_token_ids")
                        ),
                        None,
                    ),
                )
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            
            next_token_id = next_token[0, 0].item()
            self._sampling_trace.record({"stage": "selected_token", "step_index": tokens_generated, "bar_index": int(bar_index), "selected_token_id": int(next_token_id)})
            next_token_id = self.structural_token_replacements.get(
                next_token_id, next_token_id
            )

            if pending_ac_event is not None:
                pending_ac_event["first_sampled_token_after_injection"] = {
                    "id": int(next_token_id),
                    "tokens": list(decoded_token_names(self.tokenizer, next_token_id)),
                }
                pending_ac_event = None

            # Process the generated token through the model
            logits, current_state = self.model.eval(
                next_token_id, current_state, current_state, logits, use_numpy=True
            )
            logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)

            # Add to generated tokens and current sequence
            current_sequence.append(next_token_id)
            
            # Update previous tokens set for repetition penalty
            prev_tokens_set.add(next_token_id)

            #### ----------------------- TOKEN INJECTION ----------------------- ####

            did_last_token_end_in_bar_none = next_token_id in self.tokens_ending_bar_none

            if attribute_controls is not None and len(attribute_controls) > 1 and ((next_token_id in self.tokens_beginning_timesig and did_last_token_end_in_bar_none) or next_token_id in self.tokens_have_bar_none_and_timesig):
                if ac_idx >= len(attribute_controls):
                    break

                injection_index = ac_idx
                injection_tokens = [self.tokenizer.vocab[ac] for ac in attribute_controls[injection_index]]
                ac_idx += 1

                ac_event = {
                    "completed_generated_bar_index": int(injection_index - 1),
                    "next_generated_bar_index": int(injection_index),
                    "attribute_control_index": int(injection_index),
                    "attribute_control_tokens": list(attribute_controls[injection_index]),
                    "attribute_control_token_ids": [int(token_id) for token_id in injection_tokens],
                    "injection_position": int(len(current_sequence)),
                    "trigger_token": {
                        "id": int(next_token_id),
                        "tokens": list(decoded_token_names(self.tokenizer, next_token_id)),
                    },
                }

                for injected_token_id in injection_tokens:
                    logits, current_state = self.model.eval(
                        injected_token_id, current_state, current_state, logits, use_numpy=True
                    )
                    
                    # Update tracking
                    current_sequence.append(injected_token_id)
                
                # Update logits_tensor for the next iteration with the final injected token's logits
                logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
                self._ac_trace_records.append(ac_event)
                pending_ac_event = ac_event
            
            tokens_generated += 1
            
            # Check if we've generated an EOS token and can stop early
            if any(next_token_id == self.tokenizer.vocab[x] for x in ["FillBar_End", "Track_End", "EOS_None"]):
                break
        
        self._sampling_trace.close()
        if self._ac_trace_enabled and self._ac_trace_path is not None:
            with open(self._ac_trace_path, "w", encoding="utf-8") as handle:
                for record in self._ac_trace_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        # Return the complete sequence (input + generated)
        generated_tensor = torch.tensor(current_sequence, dtype=torch.long).unsqueeze(0)
        return generated_tensor
