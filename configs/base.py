from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs/train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.save_all_states = False
config.output = "/scratch365/abhatta/insightface_gauss_r100/saved_models/k3p1"


config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False
config.batch_size = 128

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False 

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 8

# WandB Logger
config.wandb_key = "e3f3c9a5016e48fabd6f8c724f2d75f1805f550e"
config.suffix_run_name = None
config.using_wandb = True
config.wandb_entity = "abhattand"
config.wandb_project = "demo-bias"
config.wandb_log_all = False
config.save_artifacts = False
config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted
