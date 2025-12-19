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
#
# This is used to build a Docker image that includes the necessary dependencies
# for running MedGemma as a microservice.

FROM python:3.12-slim AS prep

# Get pip requirements
COPY ./python/serving/requirements.txt requirements.txt
COPY ./python/serving/serving_framework/pip-install.txt pip-install.txt
RUN python3.12 -m venv --copies /payload/server-env && \
    /payload/server-env/bin/python3.12 -m pip install --require-hashes \
        -r pip-install.txt && \
    /payload/server-env/bin/python3.12 -m pip install --require-hashes \
        -r requirements.txt

# Set up code directories
COPY ./python/serving /payload/serving
COPY ./LICENSE /payload/LICENSE
RUN chmod a+x /payload/serving/entrypoint.sh

# Install git and clone source code for mirroring
RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/certifi/python-certifi.git /source-mirror/python-certifi && \
    git clone https://github.com/tqdm/tqdm.git /source-mirror/tqdm && \
    git clone https://git.launchpad.net/launchpadlib /source-mirror/launchpadlib && \
    git clone https://git.launchpad.net/lazr.restfulclient /source-mirror/lazr.restfulclient && \
    git clone https://git.launchpad.net/lazr.uri /source-mirror/lazr.uri && \
    git clone https://git.launchpad.net/wadllib /source-mirror/wadllib && \
    git clone https://gitlab.gnome.org/GNOME/pygobject.git /source-mirror/pygobject && \
    git clone https://git.launchpad.net/python-apt /source-mirror/python-apt

# ----------------------------------------------------------------------------

FROM google/cloud-sdk:stable AS gcloud_builder

# ----------------------------------------------------------------------------

FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /
# Install model dependencies for the triton backend.
COPY ./python/serving/model_requirements.txt model_requirements.txt
RUN python3 -m pip install --require-hashes -r /model_requirements.txt
# Copy in minimal gcloud install
COPY --from=gcloud_builder /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk
ENV PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

# Prevent compatibility-breaking NCCL updates.
RUN apt-mark hold libnccl-dev libnccl2 && \
    # apt upgrade
    apt-get update && \
    apt-get -y upgrade && \
    # Cleanup cached files.
    rm -rf /var/lib/apt/lists/*

# Copy the code and python environment from the prep image
COPY --from=prep /payload /
# Copy mirrored source code from the prep image
COPY --from=prep /source-mirror /source-mirror/

ENTRYPOINT ["/serving/entrypoint.sh"]
