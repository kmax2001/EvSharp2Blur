# FromS2B

TODO: Add a short project description here.

## Environment Setup

### 1. Clone Repository

```bash
git clone --recursive <REPOSITORY_URL>
cd FromS2B
```

TODO: Replace `<REPOSITORY_URL>` with the actual repository URL.

### 2. Create Environment

```bash
conda create -n froms2b python=3.8 -y
conda activate froms2b
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

TODO: Adjust the Python version if needed.
TODO: Add extra installation steps if there are system packages, CUDA, or compiler requirements.

### 3. Prepare Datasets / External Resources

TODO: Describe required datasets.
TODO: Explain where to place each dataset.
TODO: Add instructions for downloading checkpoints or pretrained models if needed.
TODO: Add any preprocessing dependency setup steps.

## Train / Test by Stage

This project appears to use multiple training and validation stages. Fill in the details for each stage below.

### Stage 1

#### Train

```bash
python tools/train_step1.py
```

TODO: Add the exact training command, config, and required arguments.
TODO: Explain input data and output checkpoints.

#### Test / Validation

```bash
python tools/valid_step1.py
```

TODO: Add the exact evaluation command and expected outputs.

### Stage 2

#### Train

```bash
python tools/train_step2.py
```

TODO: Add the exact training command, config, and required arguments.
TODO: Explain dependencies on Stage 1 outputs.

#### Test / Validation

```bash
# TODO: add stage 2 validation command
```

TODO: Add the exact evaluation command and expected outputs.

### Stage 3

#### Train

```bash
python tools/train_step3.py
```

TODO: Add the exact training command, config, and required arguments.
TODO: Explain dependencies on previous stage outputs.

#### Test / Validation

```bash
python tools/valid_step3.py
```

TODO: Add the exact evaluation command and expected outputs.

### Stage 4

#### Train

```bash
python tools/train_step4.py
```

TODO: Add the exact training command, config, and required arguments.
TODO: Explain dependencies on previous stage outputs.

#### Test / Validation

```bash
python tools/valid_step4.py
```

TODO: Add the exact evaluation command and expected outputs.

## Additional Scripts

## Results

TODO: Add benchmark tables, qualitative results, or example outputs.

## Acknowledgements

TODO: Mention external repositories, datasets, or codebases used in this project.
