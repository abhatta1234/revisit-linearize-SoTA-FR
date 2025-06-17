# revisit-linearize-SoTA-FR

Most of the code is derived from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

## training script

'config -> used rgb.py'
torchrun --nproc_per_node=4 train.py configs/${config[${SGE_TASK_ID}-1]}.py

## inference for verification protocol

python inference.py

''' correct the base path here within the inference
get the verification bin files from : 
'''

## inference for ijb protocol

python3 eval_ijbc.py \
--model-prefix ./saved_models/model.pt \
--image-path ./revisit-linearize-SoTA-FR/ijb/ijb/IJBB \
--result-dir ./revisit-linearize-SoTA-FR/ijb/results/ijbb \
--network "r100" # {r100,r18,r50} \
--job {run_name} \
--target IJBB  #{IJBB, IJBC} \
--batch-size 128

''' correct the base path here within the inference
get the verification bin files from : 
'''

## training config for adaface

To train/test with adaface - use
