# CADET-MSA

Official implementation of CADET for multimodal sentiment analysis.

## Usage

### Prerequisites

- Python 3.9.13
- PyTorch 1.13.0
- CUDA 11.7

### Installation

- Create a conda environment. Please make sure you have installed conda before.

```bash
conda create -n CADET python=3.9.13
Activate the built CADET environment.
conda activate CADET
Install PyTorch with CUDA 11.7.
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
Clone this repository.
git clone https://github.com/momoyixi/CADET-MSA.git
Install the necessary packages.
cd CADET-MSA
pip install -r requirements.txt
Datasets

This repository follows the common experimental protocol for CMU-MOSI and CMU-MOSEI in multimodal sentiment analysis.

Please put the processed datasets into the ./dataset directory and revise the corresponding paths in ./config/config.json.

For example, if the processed CMU-MOSI dataset is located in:

./dataset/MOSI/aligned_50.pkl

please make sure the corresponding configuration is set as:

"dataset_root_dir": "./dataset",
"featurePath": "MOSI/aligned_50.pkl"

Please note that raw videos and meta information are not included due to the privacy and licensing restrictions of YouTube content creators. For more details about CMU-MOSI and CMU-MOSEI, please refer to the official CMU MultimodalSDK.

Run the Codes
Training

You can first set the training dataset name in ./train.py as "mosi" or "mosei", and then run:

python train.py

By default, the trained model will be saved in the ./pt directory. You can change this path in train.py.

Testing

You can first set the testing dataset name in ./test.py as "mosi" or "mosei", and then test the trained model:

python test.py

If pretrained checkpoints are provided, please put them into the corresponding checkpoint directory and revise the path in test.py.

Notes
The dataset/ directory is not included in this repository.
The log/ directory is not included in this repository.
Please check and revise the dataset paths in ./config/config.json before running the code.
Citation information will be updated after publication.