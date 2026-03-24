from datasets import load_dataset,DatasetDict
import os

DATA_DIR="data/raw" #path

def load_squad_subset():
    dataset=load_dataset("squad") #Downloads SQuAD from HuggingFace
    
    # subset for faster training
    dataset=DatasetDict({
        "train":dataset["train"].select(range(1000)), # for training, we use a smaller subset to speed up training
        "validation":dataset["validation"].select(range(200)) # for validation, we use a smaller subset to speed up evaluation
    })
    return dataset

def save_raw_data(dataset): #saves dataset locally
    os.makedirs(DATA_DIR,exist_ok=True)
    dataset.save_to_disk(DATA_DIR)

def load_from_disk(): #loads dataset from your local folder
    from datasets import load_from_disk
    return load_from_disk(DATA_DIR)

if __name__=="__main__":
    #avoid re-downloading every time
    if os.path.exists(DATA_DIR):
        dataset = load_from_disk(DATA_DIR)
    else:
        dataset = load_squad_subset()
        save_raw_data(dataset)
    print(dataset)