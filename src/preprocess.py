from datasets import load_from_disk # load saved dataset
from transformers import AutoTokenizer # converts text → numbers
import os

# CONFIG
MODEL_NAME="bert-base-uncased" # BERT model name
MAX_LENGTH=384 # max tokens per input
STRIDE=128 # overlap when splitting long text
RAW_DATA_PATH="data/raw"
SAVE_DIR="data/processed"

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME) # Loads BERT tokenizer
# example : "What is AI?" → [101, 2054, 2003, 9932, 1029, 102]

# Loading data
def load_data():
    dataset=load_from_disk(RAW_DATA_PATH)
    return dataset["train"],dataset["validation"]

# preprocessing data
def preprocess_function(examples):
    questions=[q.strip() for q in examples["question"]] #Removes extra spaces

    inputs=tokenizer(
        questions, 
        examples["context"], # question + context
        max_length=MAX_LENGTH,
        truncation="only_second", #Only cut context (not question)
        stride=STRIDE, #If text is long:split into overlapping chunks
        return_overflowing_tokens=True, # Creating multiple chunks if needed
        return_offsets_mapping=True, #token → original text position
        padding="max_length" # makes all i/ps same size
    )

    offset_mapping=inputs["offset_mapping"] #where each token came from
    sample_mapping=inputs["overflow_to_sample_mapping"] #which original sample this chunk belongs to

    start_positions=[] # where answer starts
    end_positions=[] # where answer ends

    for i,offsets in enumerate(offset_mapping): #loop through chunks
        sample_idx=sample_mapping[i]
        answers=examples["answers"][sample_idx] #which original question this chunk belongs to

        # No answer case
        if len(answers["answer_start"])==0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # getting answer posn
        start_char=answers["answer_start"][0]
        end_char=start_char + len(answers["text"][0])
        # eg:"Delhi" → starts at char 120, ends at 125

        sequence_ids=inputs.sequence_ids(i) #identify token types
        # 0 → question , 1 → context

        # finding context start
        idx=0
        while sequence_ids[idx]!=1:
            idx+=1
        context_start=idx

        # Find context end
        while idx<len(sequence_ids) and sequence_ids[idx]==1:
            idx+=1
        context_end=idx-1

        # If answer not fully inside this chunk
        if offsets[context_start][0]>start_char or offsets[context_end][1]<end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # finding start token
            idx=context_start
            while idx<=context_end and offsets[idx][0]<=start_char:
                idx+=1
            start_positions.append(idx-1)

            # finding end token
            idx=context_end
            while idx>=context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx+1)

    inputs["start_positions"]=start_positions
    inputs["end_positions"]=end_positions
    # now model understands : answer is between token X and Y

    # Remove unnecessary fields
    inputs.pop("offset_mapping")

    return inputs

# tokenizing data
def get_tokenized_data():
    train, val = load_data()

    train = train.map(
        preprocess_function,
        batched=True,
        remove_columns=train.column_names,
    )

    val = val.map(
        preprocess_function,
        batched=True,
        remove_columns=val.column_names,
    )

    return train, val

# saving processed data
def save_processed(train, val):
    os.makedirs(SAVE_DIR, exist_ok=True)
    train.save_to_disk(f"{SAVE_DIR}/train")
    val.save_to_disk(f"{SAVE_DIR}/validation")

if __name__=="__main__":
    train,val=get_tokenized_data()
    save_processed(train,val) 

    print(train[0]) # sample


# o/p format
# {
#   "input_ids": [...],      -> word IDs from vocabulary
#   "attention_mask": [...], -> 1 for real tokens, 0 for padding
#   "token_type_ids": [...], -> 0 for question tokens, 1 for context tokens
#   "start_positions": 20,   -> index of token where answer starts
#   "end_positions": 21      -> index of token where answer ends
# }