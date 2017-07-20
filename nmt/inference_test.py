# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Tests for model inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from . import inference
from .utils import common_test_utils

float32 = np.float32
int32 = np.int32
array = np.array


class InferenceTest(tf.test.TestCase):

  def testBasicModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="",
        attention_architecture="",
        use_residual=False,)
    vocab_prefix = "nmt/testdata/test_infer_vocab"
    hparams.add_hparam("src_vocab_file", vocab_prefix + "." + hparams.src)
    hparams.add_hparam("tgt_vocab_file", vocab_prefix + "." + hparams.tgt)

    infer_file = "nmt/testdata/test_infer_file"
    out_dir = os.path.join(tf.test.get_temp_dir(), "basic_infer")
    hparams.add_hparam("out_dir", out_dir)
    os.makedirs(out_dir)
    output_infer = os.path.join(out_dir, "output_infer")
    inference.inference(out_dir, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testAttentionModel(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,)
    vocab_prefix = "nmt/testdata/test_infer_vocab"
    hparams.add_hparam("src_vocab_file", vocab_prefix + "." + hparams.src)
    hparams.add_hparam("tgt_vocab_file", vocab_prefix + "." + hparams.tgt)

    infer_file = "nmt/testdata/test_infer_file"
    out_dir = os.path.join(tf.test.get_temp_dir(), "attention_infer")
    hparams.add_hparam("out_dir", out_dir)
    os.makedirs(out_dir)
    output_infer = os.path.join(out_dir, "output_infer")
    inference.inference(out_dir, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testMultiWorkers(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=2,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,)
    vocab_prefix = "nmt/testdata/test_infer_vocab"
    hparams.add_hparam("src_vocab_file", vocab_prefix + "." + hparams.src)
    hparams.add_hparam("tgt_vocab_file", vocab_prefix + "." + hparams.tgt)

    infer_file = "nmt/testdata/test_infer_file"
    out_dir = os.path.join(tf.test.get_temp_dir(), "multi_worker_infer")
    hparams.add_hparam("out_dir", out_dir)
    os.makedirs(out_dir)
    output_infer = os.path.join(out_dir, "output_infer")

    num_workers = 3

    # There are 5 examples, make batch_size=3 makes job0 has 3 examples, job1
    # has 2 examples, and job2 has 0 example. This helps testing some edge
    # cases.
    hparams.batch_size = 3

    with tf.variable_scope("job_1"):
      inference.inference(
          out_dir, infer_file, output_infer, hparams, num_workers, jobid=1)

    with tf.variable_scope("job_2"):
      inference.inference(
          out_dir, infer_file, output_infer, hparams, num_workers, jobid=2)

    # Note: Need to start job 0 at the end; otherwise, it will block the testing
    # thread.
    with tf.variable_scope("job_0"):
      inference.inference(
          out_dir, infer_file, output_infer, hparams, num_workers, jobid=0)

    with open(output_infer) as f:
      self.assertEqual(5, len(list(f)))

  def testBasicModelWithInferIndices(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="",
        attention_architecture="",
        use_residual=False,
        inference_indices=[0])
    vocab_prefix = "nmt/testdata/test_infer_vocab"
    hparams.add_hparam("src_vocab_file", vocab_prefix + "." + hparams.src)
    hparams.add_hparam("tgt_vocab_file", vocab_prefix + "." + hparams.tgt)

    infer_file = "nmt/testdata/test_infer_file"
    out_dir = os.path.join(tf.test.get_temp_dir(), "basic_infer_with_indices")
    hparams.add_hparam("out_dir", out_dir)
    os.makedirs(out_dir)
    output_infer = os.path.join(out_dir, "output_infer")
    inference.inference(out_dir, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(1, len(list(f)))

  def testAttentionModelWithInferIndices(self):
    hparams = common_test_utils.create_test_hparams(
        encoder_type="uni",
        num_layers=1,
        attention="scaled_luong",
        attention_architecture="standard",
        use_residual=False,
        inference_indices=[1, 2])
    # TODO(rzhao): Make infer indices support batch_size > 1.
    hparams.infer_batch_size = 1
    vocab_prefix = "nmt/testdata/test_infer_vocab"
    hparams.add_hparam("src_vocab_file", vocab_prefix + "." + hparams.src)
    hparams.add_hparam("tgt_vocab_file", vocab_prefix + "." + hparams.tgt)

    infer_file = "nmt/testdata/test_infer_file"
    out_dir = os.path.join(tf.test.get_temp_dir(),
                           "attention_infer_with_indices")
    hparams.add_hparam("out_dir", out_dir)
    os.makedirs(out_dir)
    output_infer = os.path.join(out_dir, "output_infer")
    inference.inference(out_dir, infer_file, output_infer, hparams)
    with open(output_infer) as f:
      self.assertEqual(2, len(list(f)))
    self.assertTrue(os.path.exists(output_infer+str(1)+".png"))
    self.assertTrue(os.path.exists(output_infer+str(2)+".png"))


if __name__ == "__main__":
  tf.test.main()
