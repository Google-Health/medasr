#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model file compatible with Triton PyTorch 2.0 backend."""

import os
from typing import Optional

import torch
from transformers.models import lasr


# Override dir to hide imports from the Triton backend's loading strategy.
# Without this, the backend can attempt to load from the imports
# instead of MedASRWrapper.
def __dir__():
  return ["MedASRWrapper", "__name__", "__spec__"]


class MedASRWrapper(torch.nn.Module):
  """Wraps SiglipModel with custom weight loading and return structure."""

  def __init__(self):
    super(MedASRWrapper, self).__init__()
    token = None
    if os.getenv("AIP_STORAGE_URI"):
      # Using model files copied from Vertex GCS bucket.
      model_origin = os.getenv("MODEL_FILES")
    else:
      # Using model files from HF repository.
      model_origin = os.getenv("MODEL_ID")
      if not model_origin:
        raise ValueError(
            "No model origin found. MODEL_ID or AIP_STORAGE_URI must be set."
        )
      token = os.getenv("HF_TOKEN")  # optional for access to non-public models.
    self._model = lasr.LasrForCTC.from_pretrained(
        model_origin,
        token=token,
    )

  def forward(
      self,
      input_features: Optional[torch.Tensor] = None,
      attention_mask: Optional[torch.Tensor] = None,
  ):
    output = self._model.generate(
        input_features=input_features, attention_mask=attention_mask
    )
    return output
