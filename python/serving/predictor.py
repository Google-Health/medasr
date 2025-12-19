# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates embeddings for text and imaging data."""
import base64
import binascii
import io
from typing import Any, Mapping

import numpy as np
from scipy.io import wavfile
import transformers

from serving.serving_framework import model_runner
from serving.logging_lib import cloud_logging_client

# Input schema keys
AUDIO_INPUT_KEY = 'file'

# Model input dictionary mapping
MODEL_INPUT_KEYS = {
    'input_features': 'input_features__0',
    'attention_mask': 'attention_mask__1',
}

# Model output dictionary keys
TEXT_TRANCRIPT_KEY = 'text'


def load_wav(audio_string: str) -> np.ndarray:
  """Read a b64 encoded wav file for and convert to mono if needed."""
  try:
    audio_bytes = base64.b64decode(audio_string, validate=True)
  except (binascii.Error, ValueError) as exp:
    raise ValueError('Cannot decode input bytes.') from exp
  try:
    sample_rate, waveform = wavfile.read(io.BytesIO(audio_bytes))
  except ValueError as exp:
    raise ValueError('Invalid wav file.') from exp
  cloud_logging_client.debug(
      f'WAV file sample rate: {sample_rate}, shape: {waveform.shape},'
      f' dtype={waveform.dtype}'
  )
  type_info = waveform.dtype
  if waveform.ndim > 1:
    cloud_logging_client.info('Audio is stereo, converting to mono.')
    # Convert to mono by averaging the channels
    waveform = waveform.mean(axis=1)
  if sample_rate != 16000:
    raise ValueError(
        f'Sample rate {sample_rate} is not 16000, which is the expected'
        ' sample rate for audio.'
    )
  # Normalize the waveform to -1, 1 float range.
  match type_info.kind:
    case 'i':
      waveform = waveform/np.iinfo(type_info).max
    case 'u':
      raise ValueError('Unsigned wav format is not supported.')
    case 'f':
      pass  # already in -1, 1 float range.
  return waveform


class MedASRPredictor:
  """Callable responsible for generating embeddings."""

  def __init__(self, model_source: str, token: str | None = None):
    self._model_source = model_source
    self._token = token
    self._processor = None

  def predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs inference on provided patches.

    Args:
      prediction_input: JSON formatted input for embedding prediction.
      model: ModelRunner to handle model step.

    Returns:
      JSON formatted output.
    """

    if self._processor is None:
      self._processor = transformers.AutoProcessor.from_pretrained(
          self._model_source,
          token=self._token,
      )

    # build response for each instance.
    try:
      audio = load_wav(prediction_input[AUDIO_INPUT_KEY])
      cloud_logging_client.debug('Audio loaded.')
      processed = self._processor(audio, return_tensors='np')
      cloud_logging_client.debug('Model input processed.')
      model_input = {
          MODEL_INPUT_KEYS[key]: value for key, value in processed.items()
      }
      tokens = model.run_model(model_input, model_output_key='tokens__0')
      cloud_logging_client.debug('Model run completed.')
      transcript = self._processor.batch_decode(tokens)[0]
      cloud_logging_client.debug('Tokens decoded.')
      cloud_logging_client.info('Returning transcripts.')
      return{TEXT_TRANCRIPT_KEY: transcript}
    except ValueError as exp:
      cloud_logging_client.warning('Failed loading wav file.')
      return {'error': str(exp)}
