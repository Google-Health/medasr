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

"""Tests for MedASR predictor."""

import base64

from absl.testing import absltest
from absl.testing import parameterized

from serving import predictor


class PredictorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'not_base64',
          'audio_string': 'not_base64',
      },
      {
          'testcase_name': 'not_wav',
          'audio_string': base64.b64encode(b'not_wav').decode('utf-8'),
      },
  )
  def test_bad_file_loading_raises_error(self, audio_string: str):
    with self.assertRaises(ValueError):
      predictor.load_wav(audio_string)


if __name__ == '__main__':
  absltest.main()
