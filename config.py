import os
import torch

# 随机种子
SEED = 42

# 路径配置
ROOT_PATH = os.path.join('/', 'home', 'adduser', 'Shenji', 'UA-TRAC')
IMG_ROOT = os.path.join(ROOT_PATH, 'Insight-MVT_Annotation_Train')
ORIGINAL_GT_PATH = os.path.join(ROOT_PATH, 'train_gt.txt')
OUTPUT_IMG_DIR = os.path.join(ROOT_PATH, 'outputting_imgs')
NEW_GT_DIR = os.path.join(ROOT_PATH, 'new_gt_files')

# 训练参数
TEST_RATIO = 0.1
VAL_RATIO = 0.1
EPOCHS = 5
BATCH_SIZE = 8
MAX_BOXES = 30
IMG_SIZE = 640
NUM_CLASSES = 1
MAX_SAMPLES = None

# 优化器参数
LR = 0.2
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
ACCUMULATE_STEPS = 2

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')