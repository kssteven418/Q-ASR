# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
Code referred from NeMo/examples/asr/speech_to_text_infer.py
"""

from argparse import ArgumentParser

import torch

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.utils import logging
import nemo.quantization.utils.quantize_model as qm
from nemo.quantization.utils.distill_data import *

import sys
import os
from tqdm import tqdm 
import pickle

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'")
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--num_batch", type=int, default=50, help="number of batches of the synthetic data")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size of the synthetic data")
    parser.add_argument("--seqlen", type=int, default=500, help="sequence length of the synthetic data")
    parser.add_argument("--train_iter", type=int, default=200, help="training iterations for the synthetic data generation")
    parser.add_argument("--dump_path", type=str, default=None, help="path to dump the synthetic data")
    parser.add_argument("--dump_prefix", type=str, default='syn', help="prefix for the filename of the dumped synthetic data")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the synthetic data generation")

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        teacher_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        teacher_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)

    teacher_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': teacher_model.decoder.vocabulary,
            'batch_size': 8,
            'normalize_transcripts': True,
            'shuffle': True,
        }
    )

    ############################## Distillation #####################################

    teacher_model.set_quant_mode('none') # distable quantization mode for the teacher model
    torch.set_grad_enabled(True) # enable backward graph generation

    print("Num batches: %d, Batch size: %d, Training iterations: %d, Learning rate: %.3f " \
            % (args.num_batch, args.batch_size, args.train_iter, args.lr))
    print('Synthesizing...')

    synthetic_data = get_synthetic_data(teacher_model.encoder, teacher_model.decoder, batch_size=args.batch_size, 
            dim=64, seqlen=args.seqlen, num_batch=args.num_batch, train_iter=args.train_iter, lr=args.lr)

    file_name = '%s_nb%d_iter%d_lr%.3f.pkl' % \
            (args.dump_prefix,  args.num_batch, args.train_iter, args.lr)

    if args.dump_path is not None:
        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        file_name = os.path.join(args.dump_path, file_name)

    print('Synthetic data dumped as ', file_name)
    with open(file_name, 'wb') as f:
        pickle.dump([x.cpu() for x in synthetic_data], f)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
