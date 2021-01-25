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
From NeMo/examples/asr/speech_to_text_infer.py
Usage: 
    python inference.py \
    --asr_model /rscratch/data/librispeech/nemo_models/QuartzNet15x5Base-En.nemo  \
    --dataset /rscratch/data/librispeech/dev_other.json  

You can also specify `--distill_dump [filename]` to store the distilled data
Then, you can load the distilled data by `--distill_load [path]` to avoid distillation stage for the next runs

By default, distillation is enabled and is executed with:
    * training iteration = 300
    * batch size = 8
    * number of batches = 50 (thereby 8 * 50 = 400 data in total)
You can change this dafault setting by specifying e.g., `--distill_train_iter 100  --batch_size 4 --distill_num_batch 10`

Finally, you can enable dynamic quantization mode by specifying `--dynamic`
This will skip distillation and calibration stages as they are unnecessary
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'")
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument("--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English.")
    parser.add_argument("--store_model", type=str, default=None, help="path to store the model")
    parser.add_argument("--cnt", type=int, default=None, help="early stop for debugging")

    parser.add_argument("--dynamic", action='store_true', help="Dynamic quantization mode.")
    parser.add_argument("--normalize", action='store_true', help="Normalize calib data.")

    parser.add_argument("--distill_num_batch", type=int, default=50, help="number of batches for distilled data")
    parser.add_argument("--distill_train_iter", type=int, default=200, help="training iteration for distilled data generation")
    parser.add_argument("--distill_batch_size", type=int, default=8)
    parser.add_argument("--distill_dump", type=str, default=None, help="Pickle dump filename for distilled data")
    parser.add_argument("--distill_load", type=str, default=None, help="Pickle load path for distilled data")
    parser.add_argument("--alpha", type=float, default=0.0, help="regularization constant for distillation") 
    parser.add_argument("--beta", type=float, default=0.0, help="regularization constant for distillation")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    if args.store_model is not None and not args.store_model.endswith('.nemo'):
        raise Exception('--store_model: file name must end with .nemo')

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
        teacher_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)
        teacher_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)

    for _model in [asr_model, teacher_model]:
        _model.setup_test_data(
            test_data_config={
                'sample_rate': 16000,
                'manifest_filepath': args.dataset,
                'labels': asr_model.decoder.vocabulary,
                'batch_size': args.batch_size,
                'normalize_transcripts': args.normalize_text,
            }
        )

    if args.store_model is not None:
        asr_model.save_to(args.store_model)
        print('model stored at %s' % args.store_model)
        sys.exit()


    if not args.dynamic:
        teacher_model.set_quant_mode('none') # distable quantization mode for the teacher model
        torch.set_grad_enabled(True) # enable backward graph generation
        distill_seqlen = 500

        # if load path for the distilled data is given, skip the data distillation process.
        if args.distill_load is not None:
            print('data loaded from %s' % args.distill_load)
            with open(args.distill_load, 'rb') as f:
                distilled_data = pickle.load(f)
        else:
            print('distillating')
            distilled_data = get_distill_data(teacher_model.encoder, teacher_model.decoder, batch_size=args.distill_batch_size, 
                    dim=64, seqlen=distill_seqlen, num_batch=args.distill_num_batch, train_iter=args.distill_train_iter,
                    alpha=args.alpha, beta=args.beta)
            # if dump path is given, dump the distilled data
            if args.distill_dump is not None:
                file_name = '%s-%d-%d-a%f.pkl' % (args.distill_dump, args.distill_num_batch, args.distill_train_iter, args.alpha)
                print('model dumped as ', file_name)
                with open(file_name, 'wb') as f:
                    pickle.dump(distilled_data, f)
                with open('cpu_'+file_name, 'wb') as f:
                    pickle.dump([x.cpu() for x in distilled_data], f)

    torch.set_grad_enabled(False) # disable backward graph generation
    asr_model.eval() # evaluation mode
    asr_model.encoder.bn_folding() # BN folding

    # if not dynamic quantization mode, calibrate min/max/range for the activations using the distilled data
    if not args.dynamic:
        print('calibrating')
        qm.calibrate(asr_model)
        length = torch.tensor([distill_seqlen] * args.distill_batch_size).cuda()
        for batch_idx, inputs in enumerate(distilled_data):
            inputs = inputs.cuda()
            if args.normalize:
                m = inputs.mean(axis=-1, keepdim=True)
                s = inputs.std(axis=-1, keepdim=True)
                inputs = (inputs - m) / s
                inputs = torch.clamp(inputs, min=-4, max=4)
            encoded, encoded_len, encoded_scaling_factor = asr_model.encoder(audio_signal=inputs, length=length)
            log_probs = asr_model.decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)

    print('evaluating')
    qm.evaluate(asr_model)
    qm.adjust_range(asr_model, 1.1) # adjust min/max to 1.1*min/1.1*max

    qm.set_dynamic(asr_model, args.dynamic) # if dynamic quantization mode, this will enabled
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    wer = WER(vocabulary=asr_model.decoder.vocabulary)
    hypotheses = []
    references = []
    progress_bar = tqdm(asr_model.test_dataloader())
    cnt = 0

    for i, test_batch in enumerate(progress_bar):
        cnt += 1
        if cnt == args.cnt:
            break

        if can_gpu:
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
    if wer_value > args.wer_tolerance:
        raise valueerror(f"got wer of {wer_value}. it was higher than {args.wer_tolerance}")
    logging.info(f'got wer of {wer_value}. tolerance was {args.wer_tolerance}')
    print('value:', wer_value)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
