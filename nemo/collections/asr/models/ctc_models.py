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
import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import onnx
import torch
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.utils.export_utils import attach_onnx_to_onnx
import nemo.quantization.utils.quantize_model as quantize_model

__all__ = ['EncDecCTCModel', 'JasperNet', 'QuartzNet']


class EncDecCTCModel(ASRModel, Exportable):
    """Base class for encoder decoder CTC-based models."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5Base-En",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo",
            description="QuartzNet15x5 model trained on six datasets: LibriSpeech, Mozilla Common Voice (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. It was trained with Apex/Amp optimization level O1 for 600 epochs. The model achieves a WER of 3.79% on LibriSpeech dev-clean, and a WER of 10.05% on dev-other.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5Base-Zh",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-Zh.nemo",
            description="QuartzNet15x5 model trained on ai-shell2 Mandarin Chinese dataset.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet5x5LS-En",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet5x5LS-En.nemo",
            description="QuartzNet5x5 model trained on LibriSpeech dataset only. The model achieves a WER of 5.37% on LibriSpeech dev-clean, and a WER of 15.69% on dev-other.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5NR-En",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5NR-En.nemo",
            description="QuartzNet15x5Base-En was finetuned with RIR and noise augmentation to make it more robust to noise. This model should be preferred for noisy speech transcription. This model achieves a WER of 3.96% on LibriSpeech dev-clean and a WER of 10.14% on dev-other.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="Jasper10x5Dr-En",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/Jasper10x5Dr-En.nemo",
            description="JasperNet10x5Dr model trained on six datasets: LibriSpeech, Mozilla Common Voice (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. It was trained with Apex/Amp optimization level O1. The model achieves a WER of 3.37% on LibriSpeech dev-clean, 9.81% on dev-other.",
        )
        result.append(model)
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.enable_spec_augmentation = False
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecCTCModel.from_config_dict(self._cfg.preprocessor)
        self.quant_mode = 'symmetric'
        with omegaconf.open_dict(self._cfg.encoder):
            self._cfg.encoder.quant_mode = 'symmetric'
        with omegaconf.open_dict(self._cfg.decoder):
            self._cfg.decoder.quant_mode = 'symmetric'
        self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)

        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup metric objects
        self._wer = WER(
            vocabulary=self.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def set_quant_mode(self, quant_mode):
        self.encoder.set_quant_mode(quant_mode)
        self.decoder.set_quant_mode(quant_mode)


    @torch.no_grad()
    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, logprobs=False) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            paths2audio_files: (a list) of paths to audio files. \
        Recommended length per file is between 5 and 25 seconds. \
        But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
        Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.

        Returns:

            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}
        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in temporary_datalayer:
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            hypotheses.append(logits[idx][: logits_len[idx]])
                    else:
                        hypotheses += self._wer.ctc_decoder_predictions_tensor(greedy_predictions)
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            logging.set_verbosity(logging_level)
        return hypotheses

    def change_vocabulary(self, new_vocabulary: List[str]):
        """
        Changes vocabulary used during CTC decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        If new_vocabulary == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
            this is target alphabet.

        Returns: None

        """
        if self.decoder.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.decoder.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['vocabulary'] = new_vocabulary
            new_decoder_config['num_classes'] = len(new_vocabulary)

            del self.decoder
            self.decoder = EncDecCTCModel.from_config_dict(new_decoder_config)
            del self.loss
            self.loss = CTCLoss(
                num_classes=self.decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self._cfg.get("ctc_reduction", "mean_batch"),
            )
            self._wer = WER(
                vocabulary=self.decoder.vocabulary,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                ctc_decode=True,
                dist_sync_on_step=True,
                log_prediction=self._cfg.get("log_prediction", False),
            )

            # Update config
            OmegaConf.set_struct(self._cfg.decoder, False)
            self._cfg.decoder = new_decoder_config
            OmegaConf.set_struct(self._cfg.decoder, True)

            logging.info(f"Changed decoder to output to {self.decoder.vocabulary} vocabulary.")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_char_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and (self.training or self.enable_spec_augmentation):
            processed_signal = self.spec_augmentation(input_spec=processed_signal)

        encoded, encoded_len, encoded_scaling_factor = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions, targets=transcript, target_lengths=transcript_len,
            )
            wer, _, _ = self._wer.compute()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        quantize_model.evaluate(self)
        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        self._wer.update(predictions=predictions, targets=transcript, target_lengths=transcript_len)
        wer, wer_num, wer_denom = self._wer.compute()
        quantize_model.train(self)

        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_loss': logs['val_loss'],
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            'test_wer': logs['val_wer'],
        }
        return test_logs

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.decoder.vocabulary,
            'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
            'trim_silence': True,
            'shuffle': False,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 12,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = True,
        use_dynamic_axes: bool = True,
    ):
        if input_example is not None or output_example is not None:
            logging.warning(
                "Passed input and output examples will be ignored and recomputed since"
                " EncDecCTCModel consists of two separate models (encoder and decoder) with different"
                " inputs and outputs."
            )

        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        output1 = os.path.join(os.path.dirname(output), 'encoder_' + os.path.basename(output))
        output1_descr = qual_name + ' Encoder exported to ONNX'
        encoder_onnx = self.encoder.export(
            output1,
            None,  # computed by input_example()
            None,
            verbose,
            export_params,
            do_constant_folding,
            keep_initializers_as_inputs,
            onnx_opset_version,
            try_script,
            set_eval,
            check_trace,
            use_dynamic_axes,
        )

        output2 = os.path.join(os.path.dirname(output), 'decoder_' + os.path.basename(output))
        output2_descr = qual_name + ' Decoder exported to ONNX'
        decoder_onnx = self.decoder.export(
            output2,
            None,  # computed by input_example()
            None,
            verbose,
            export_params,
            do_constant_folding,
            keep_initializers_as_inputs,
            onnx_opset_version,
            try_script,
            set_eval,
            check_trace,
            use_dynamic_axes,
        )

        output_model = attach_onnx_to_onnx(encoder_onnx, decoder_onnx, "DC")
        output_descr = qual_name + ' Encoder+Decoder exported to ONNX'
        onnx.save(output_model, output)
        return ([output, output1, output2], [output_descr, output1_descr, output2_descr])


class JasperNet(EncDecCTCModel):
    pass


class QuartzNet(EncDecCTCModel):
    pass
