MODEL_NAME = "Locutusque/TinyMistral-248M"
DATASET_NAME = "databricks/databricks-dolly-15k"
TEACHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4

OUTPUT_DIR_S = "./outputs/dolly-sft-soft"
OUTPUT_DIR = "./outputs/dolly-sft"
SSFT_MODEL_PATH = "./outputs/dolly-sft-soft"
SFT_MODEL_PATH = "./outputs/dolly-sft"