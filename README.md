# FromS2B

## TODO

- [ ] Upload a training/test datasets and checkpoints.
- [ ] Upload data preprocessing details.


## Environment Setup

### 1. Clone Repository

```bash
git clone --recursive https://github.com/kmax2001/EvSharp2Blur.git
cd FromS2B
```

### 2. Create Environment

```bash
conda create -n froms2b python=3.8 -y
conda activate froms2b
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### 3. Prepare Datasets / External Resources

## Train / Test by Stage

### Stage 1

#### Train

```bash
python tools/train.py --cfg experiments/crowdpose/w32/w_32_train_event_subteacher.yaml

python tools/train.py --cfg experiments/crowdpose/w32/w_32_train_blur2blur_subteacher.yaml

python tools/train_step1.py --cfg experiments/crowdpose/w32/w_32_train_step1_blur2blur_subteacher.yaml
```

#### Test / Validation

```bash
python tools/valid_step1.py --cfg experiments/crowdpose/w32/w_32_test_event_subteacher.yaml

python tools/valid_step1.py --cfg experiments/crowdpose/w32/w_32_test_blur2blur_subteacher.yaml

python tools/valid_step1.py --cfg experiments/crowdpose/w32/w_32_test_step1_blur2blur_subteacher.yaml
```

### Stage 2

#### Train

```bash
python tools/train_step2.py --cfg experiments/crowdpose/w32/w_32_train_step2.yaml
```

#### Test / Validation

```bash
python tools/valid_step2.py --cfg experiments/crowdpose/w32/w_32_test_step2.yaml
```

### Stage 3

#### Train

```bash
python tools/train_step3.py --cfg experiments/crowdpose/w32/w_32_train_step3.yaml
```

#### Test / Validation

```bash
python tools/valid_step3.py --cfg experiments/crowdpose/w32/w_32_test_step3.yaml
```

### Stage 4

#### Train

```bash
python tools/train_step4.py --cfg experiments/crowdpose/w32/w_32_train_step4.yaml
```

#### Test / Validation

```bash
python tools/valid_step4.py --cfg experiments/crowdpose/w32/w_32_test_step4.yaml
```
