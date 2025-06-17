# Revisit Linearize State-of-the-Art Face Recognition

This repository contains implementations for face recognition models, with most of the code derived from the [InsightFace ArcFace implementation](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).

## Training Script

### Configuration
Use the RGB configuration file located in the `config` directory:

```bash
# Training command with distributed setup
torchrun --nproc_per_node=4 train.py configs/{config_name}.py
```

**Note:** For original training `rgb.py` from the config directory was used for hyperparameter.

## Inference for Verification Protocol

Run the verification inference script:

```bash
python inference.py
```

> **Important:** Correct the base path within the inference script and obtain the verification bin files from the appropriate source.

## Inference for IJB Protocol

Execute IJB evaluation with the following command:

```bash
python3 eval_ijbc.py \
    --model-prefix ./saved_models/model.pt \
    --image-path ./revisit-linearize-SoTA-FR/ijb/ijb/IJBB \
    --result-dir ./revisit-linearize-SoTA-FR/ijb/results/ijbb \
    --network "r100" \
    --job {run_name} \
    --target IJBB \
    --batch-size 128
```

### Parameter Options:
- `--network`: Choose from `{r100, r18, r50}`
- `--target`: Choose from `{IJBB, IJBC}`
- `--job`: Specify your run name

> **Important:** Correct the base path within the inference script and obtain the verification bin files from the appropriate source.

## Training Configuration for AdaFace

### AdaFace Training/Testing
To train or test with AdaFace, use the backbone with Global Weighted Pooling (GWP):
- **Backbone:** `backbones/adaface_backbone`
- **Integration:** This can be plugged into the [AdaFace repository](https://github.com/mk-minchul/AdaFace)

### Non-Adaptive Pool Training/Testing  
To train or test with Adaptive Pool, use the backbone without attention:
- **Backbone:** `backbones/iresenet_gap.py`
- **Description:** Sample code for adaptive pooling without attention mechanism

## Dependencies

Make sure you have the required dependencies installed. Refer to the original InsightFace repository for detailed installation instructions.
The only thing that changes is the backbone.

## Usage Notes

1. Ensure all file paths are correctly configured before running inference scripts
2. Download the necessary verification bin files/ijb test files from the appropriate sources
> **Important:** https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
3. Adjust batch size and other parameters based on your hardware capabilities
