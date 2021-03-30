# Q-ASR: Integer-only Zero-shot Quantization for Efficient Speech Recognition
---
<img width="900" alt="zsq_horiz3" src="https://user-images.githubusercontent.com/50283958/113030937-a29bef80-917d-11eb-8f33-65fd5b076d0e.png">

## 1. Installation and Requirements
You can find detailed installation guides from the [NeMo repo](https://github.com/NVIDIA/NeMo).

1. Create a Conda virtual environment
```
conda create -n qasr python=3.8
conda activate qasr
```

2. Install requirements
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install Cython
```

3. Install NeMo (Q-ASR) 
```
git clone https://github.com/kssteven418/Q-ASR.git
cd Q-ASR
./reinstall.sh
```


## 2. Datasets Download and Preprocessing
Q-ASR is evaluated on the Librispeech dataset, which can be downloaded and preprocessed using the script provided by NeMo. 
You can find the script in `Q-ASR/scripts/get_librispeech_data.py`. 
Run the script using the following command.
```
# in Q-ASR/scripts
python get_librispeech_data.py --data_sets {dataset} --data_root {DIR}
```
`{datasets}` can be one of the following: `{dev_clean, dev_other, train_clean_100, train_clean_360, train_other_500, test_clean, test_other}`.
You can also concatenate multiple items with comma(,) to process multiple datasets (e.g., `dev_clean,dev_other`),
or use `ALL` to process all.

After processing `dev_clean`, for example, the preprocessed datasets will be stored at `{DIR}/LibriSpeech/dev-clean-processed`. 
Additionally, a manifest file is generated at `{DIR}/dev_clean.json`. 
This is a json file that links the preprocessed audio files in `{DIR}/LibriSpeech/dev-clean-processed` with the corresponding text labels.
Therefore, make sure not to move the preprocessed audio files to another directory unless you modify the manifest file accordingly 
(otherwise, the manifest file will not locate the audio files).


## 3. Run Q-ASR
Q-ASR consists of 2 steps: (1) Synthetic data generation, and (2) Calibration and evaluation, each of which can be run with the python scripts
`synthesize.py` and `inference.py` in `Q-ASR/examples/asr/quantization`.

### 3-1. Synthetic Data Generation
Run the following command for synthetic data generation.
```
# in Q-ASR/examples/asr/quantization
python synthesize.py --asr_model {model_name} --dataset {path_to_manifest} \
                     --num_batch {num_batch} --batch_size {batch_size} \
                     --seq_len {seq_len} --train_iter {train_iter} --lr {lr} \
                     --dump_path {dump_path} --dump_prefix {dump_prefix}
```

For instance,
```
python synthesize.py --asr_model QuartzNet15x5Base-En --dataset {DIR}/dev_clean.json \
                     --num_batch 20 --batch_size 8 \
                     --seq_len 500 --train_iter 200 --lr 0.05 \
                     --dump_path dump --dump_prefix quartznet
```

Note that `{DIR}/dev_clean.json` is the manifest file (generated from the preprocessing step) for the target evaluation dataset.
Please use the flag `-h` to see the details for the input arguments.
The resulting dataset is stored at `{dump_path}` and will be loaded in the following calibration/evaluation step.

### 3-2. Calibration/Evaluation
After generating the synthetic data, run the following command to calibrate and evaluate the quantized model.
```
# in Q-ASR/examples/asr/quantization
python inference.py --asr_model {model_name} --dataset {path_to_manifest} \
                    --load {load} --weight_bit {wb} --act_bit {ab}  --percentile {p}
```
For instance,
```
python inference.py --asr_model QuartzNet15x5Base-En --dataset {DIR}/dev_clean.json \
                    --load dump/quartznet_nb20_iter200_lr0.050.pkl \
                    --weight_bit 6 --act_bit 6  --percentile 99.996
```
Similarly, `{DIR}/dev_clean.json` is the manifest file (generated from the preprocessing step) for the target evaluation dataset.
Please use the flag `-h` to see the details for the input arguments.
We can also use `--dynamic` flag to perform dynamic quantization, instead of using `--load` flag to load the synthetic dataset.
