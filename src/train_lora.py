import time
from datasets import load_from_disk
from transformers import(
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model
import evaluate

#Config
MODEL_NAME="bert-base-uncased"
DATA_PATH="data/processed"
MODEL_SAVE_PATH="models/lora"

BATCH_SIZE=16
EPOCHS=1
LEARNING_RATE=3e-4

# loading data
def load_data():
    train=load_from_disk(f"{DATA_PATH}/train")
    val=load_from_disk(f"{DATA_PATH}/validation")
    return train, val

# loading model and applying LoRA
def load_lora_model():
    model=AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    lora_config=LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    model=get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# metrics
def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    start_preds,end_preds=predictions
    start_labels,end_labels=labels
    correct=0
    total=len(start_preds)

    for i in range(total):
        if start_preds[i].argmax()==start_labels[i] and end_preds[i].argmax()==end_labels[i]:
            correct+=1

    accuracy=correct/total
    return {"accuracy": accuracy}

# training
def train():
    train_data, val_data=load_data()
    model=load_lora_model()

    data_collator=DefaultDataCollator()
    training_args=TrainingArguments(output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        report_to="none"
    )
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("training LoRA")
    start_time=time.time()
    trainer.train()
    end_time=time.time()
    training_time=end_time-start_time
    print(f"Training time: {training_time:.2f} seconds")
    model.save_pretrained(MODEL_SAVE_PATH)
    return training_time

if __name__=="__main__":
    train()