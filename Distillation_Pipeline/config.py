TEACHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
STUDENT_MODEL_S = "../Baseline/outputs/tinymistral-sft-merged-soft"
STUDENT_MODEL = "../Baseline/outputs/tinymistral-sft-merged"
DATASET_NAME = "databricks/databricks-dolly-15k"

MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
EPOCHS = 7
LR = 1e-4
TEMPERATURE = 4.5
ALPHA = 0.9

DISTILLED_OUTPUT_DIR = "./outputs/logit-distilled-tinymistral"
DISTILLED_OUTPUT_DIR_S = "./outputs/logit-distilled-tinymistral-Soft"