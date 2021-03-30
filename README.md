# Q-ASR: Integer-only Zero-shot Quantization for Efficient Speech Recognition
---

## Installation and Requirements
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

## Dataset Download and Preprocessing
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
