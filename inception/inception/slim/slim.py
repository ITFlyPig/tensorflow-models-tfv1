# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TF-Slim grouped API. Please see README.md for details and usage."""
import tensorflow as tf

# pylint: disable=unused-import

# Collapse tf-slim into a single namespace.
from inception.slim import inception_model as inception
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import ops
from tensorflow.contrib.slim import scopes
from tensorflow.contrib.slim import variables
from tensorflow.contrib.slim import arg_scope
