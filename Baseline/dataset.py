from datasets import load_dataset
from config import DATASET_NAME


def load_dolly():
    dataset = load_dataset(DATASET_NAME)

    dataset = dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    def valid_example(x):
        return (
            x["instruction"] is not None and
            x["response"] is not None and
            len(x["response"].strip()) > 10
        )

    train_dataset = train_dataset.filter(valid_example)
    val_dataset = val_dataset.filter(valid_example)

    train_dataset = train_dataset.select(range(10000))

    return train_dataset, val_dataset
