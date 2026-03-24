#data/processed → train + validation
import time
import os
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
import evaluate

# config
MODEL_NAME="bert-base-uncased"
DATA_PATH="data/processed"
MODEL_SAVE_PATH="models/baseline"
BATCH_SIZE=16
EPOCHS=1
LEARNING_RATE=3e-5

# loading data
def load_data():
    train=load_from_disk(f"{DATA_PATH}/train")
    val=load_from_disk(f"{DATA_PATH}/validation")
    return train,val

# loading model
def load_model():
    model=AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return model

metric=evaluate.load("squad")

def compute_metrics(eval_pred):
    predictions,labels=eval_pred
    start_preds,end_preds=predictions
    start_labels,end_labels=labels

    #Simplified evaluation(token-level)
    correct=0
    total=len(start_preds)

    for i in range(total):
        if start_preds[i].argmax()==start_labels[i] and end_preds[i].argmax()==end_labels[i]:
            correct+=1
        
    accuracy=correct/total
    return {"accuracy":accuracy}

# training
def train():
    train_data,val_data=load_data()
    model=load_model()

    data_collator=DefaultDataCollator()

    training_args=TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("training starting")
    start_time=time.time()
    trainer.train()
    end_time=time.time()
    training_time=end_time-start_time

    print(f"Training time:{training_time:.2f} seconds")

    print("Saving model")
    trainer.save_model(MODEL_SAVE_PATH)
    return training_time

if __name__=="__main__":
    training_time=train()