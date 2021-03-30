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
if not can_gpu:
    raise Exception("Current implementation only supports GPU")

def main():
    parser = ArgumentParser()

    """Training arguments"""
    parser.add_argument("--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'")
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English.")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle test data.")

    """Calibration arguments"""
    parser.add_argument("--load", type=str, default=None, help="load path for the synthetic data")
    parser.add_argument("--percentile", type=float, default=None, help="Max/min percentile for outlier handling. e.g., 99.9")

    """Quantization arguments"""
    parser.add_argument("--weight_bit", type=int, default=8, help="quantization bit for weights")
    parser.add_argument("--act_bit", type=int, default=8, help="quantization bit for activations")
    parser.add_argument("--dynamic", action='store_true', help="Dynamic quantization mode.")
    parser.add_argument("--no_quant", action='store_true', help="No quantization mode.")

    """Debugging arguments"""
    parser.add_argument("--eval_early_stop", type=int, default=None, help="early stop for debugging")
    parser.add_argument("--calib_early_stop", type=int, default=None, help="early stop calibration")

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)

    
    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': args.normalize_text,
            'shuffle': args.shuffle,
        }
    )

    if args.load is not None:
        print('Data loaded from %s' % args.load)
        with open(args.load, 'rb') as f:
            distilled_data = pickle.load(f)
        synthetic_batch_size, _, synthetic_seqlen = distilled_data[0].shape
    else:
        assert args.dynamic, \
                "synthetic data must be loaded unless running with the dynamic quantization mode"

    
    ############################## Calibration #####################################

    torch.set_grad_enabled(False) # disable backward graph generation
    asr_model.eval() # evaluation mode
    asr_model.set_quant_bit(args.weight_bit, mode='weight')
    asr_model.set_quant_bit(args.act_bit, mode='act')

    # set percentile
    if args.percentile is not None:
        qm.set_percentile(asr_model, args.percentile)

    if args.no_quant:
        asr_model.set_quant_mode('none')
    else:
        asr_model.encoder.bn_folding() # BN folding

    # if not dynamic quantization, calibrate min/max/range for the activations using synthetic data
    # if dynamic, we can skip calibration
    if not args.dynamic:
        print('Calibrating...')
        qm.calibrate(asr_model)
        length = torch.tensor([synthetic_seqlen] * synthetic_batch_size).cuda()
        for batch_idx, inputs in enumerate(distilled_data):
            if args.calib_early_stop is not None and batch_idx == args.calib_early_stop:
                break
            inputs = inputs.cuda()
            encoded, encoded_len, encoded_scaling_factor = asr_model.encoder(audio_signal=inputs, length=length)
            log_probs = asr_model.decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)


    ############################## Evaluation  #####################################

    print('Evaluating...')
    qm.evaluate(asr_model)

    qm.set_dynamic(asr_model, args.dynamic) # if dynamic quantization, this will be enabled
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    wer = WER(vocabulary=asr_model.decoder.vocabulary)
    hypotheses = []
    references = []
    progress_bar = tqdm(asr_model.test_dataloader())

    for i, test_batch in enumerate(progress_bar):
        if i == args.eval_early_stop:
            break
        test_batch = [x.cuda().float() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(greedy_predictions.shape[0]):
            reference = ''.join([labels_map[c] for c in test_batch[2][batch_ind].cpu().detach().numpy()])
            references.append(reference)
        del test_batch
    wer_value = word_error_rate(hypotheses=hypotheses, references=references)
    print('WER:', wer_value)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
