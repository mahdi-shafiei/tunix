# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM rollout worker with Tunix sampler."""

from typing import Any, Dict, Optional
from flax import nnx
import jax
import jaxtyping
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


class VllmRollout(base_rollout.BaseRollout):
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      lora_config: Optional[Dict[str, str]] = None,
      model_version: str = "meta-llama/Llama-3.1-8B",
  ):
    self.mesh = mesh
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        max_model_len=cache_config_or_size,
        mesh=mesh,
        model_version=model_version,
        hbm_utilization=0.3,
        mapping_config=vllm_sampler.MappingConfig(
            to_hf_mappings=model.to_hf_mappings(),
            to_hf_transpose_keys=model.to_hf_transpose_keys(),
            lora_to_hf_mappings=model.lora_to_hf_mappings(),
            lora_config=lora_config,
        ),
    )
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    self.output = self._sampler(
        input_strings=prompts,
        total_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=kwargs.get("max_prompt_length", None),
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        echo=False,
        pad_output=kwargs.get("max_prompt_length", True),
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    # b/428730696, we cannot return self.output.logprobs yet
    # May need to validate if there will be any difference from recalculation
    return self.output.logprobs

  def update_params(self, params: jaxtyping.PyTree) -> None:
    self._sampler.update_params(params)

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer
