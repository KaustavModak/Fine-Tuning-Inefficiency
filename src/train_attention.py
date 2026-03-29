import time
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/attention"

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 3e-5


# loading preprocessed datasets from disk
def load_data():
    train = load_from_disk(f"{DATA_PATH}/train")
    val = load_from_disk(f"{DATA_PATH}/validation")
    return train, val


# loading model and freezing all layers except attention
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    #  Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    #  Unfreeze only attention layers
    for name, param in model.named_parameters():
        if "attention" in name:
            param.requires_grad = True

    #  Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"% Trainable: {100 * trainable / total:.4f}")

    return model


# metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    start_preds, end_preds = predictions
    start_labels, end_labels = labels

    correct = 0
    total = len(start_preds)

    for i in range(total):
        if start_preds[i].argmax() == start_labels[i] and end_preds[i].argmax() == end_labels[i]:
            correct += 1

    return {"accuracy": correct / total}


# training function
def train():
    train_data, val_data = load_data()

    print(" Loading model (Attention Only)...")
    model = load_model()

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
    )

    print(" Training Attention-Only Model...")

    start = time.time()
    trainer.train()
    end = time.time()

    print(f"Training Time: {end - start:.2f} seconds")

    model.save_pretrained(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()