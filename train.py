import numpy as np
from tensorflow import keras
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import keras_nlp
from transformers.keras_callbacks import KerasMetricCallback
from rouge_score import rouge_scorer

rouge_l = keras_nlp.metrics.RougeL()


TRAIN_TEST_SPLIT = 0.1

MAX_INPUT_LENGTH = 256  # Maximum length of the input to the model
MIN_TARGET_LENGTH = 5  # Minimum length of the output by the model
MAX_TARGET_LENGTH =16 # Maximum length of the output by the model
BATCH_SIZE = 8  # Batch-size for training the model
LEARNING_RATE = 2e-5  # Learning-rate for training the model
MAX_EPOCHS = 5  # Maximum number of epochs the model is trained for


MODEL_CHECKPOINT = "t5-base"



def train():
    
    
    #load dataset, change path accoring to your need
    dataset = load_dataset('csv', data_files='train.csv',split="train")
    
    #convert dataset into huggingface format
    raw_datasets =dataset.train_test_split(train_size=0.9, test_size=TRAIN_TEST_SPLIT)
    
    #load tokenizer model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    if MODEL_CHECKPOINT in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
        
    #preprocessing data
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True,padding=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["title"], max_length=16, truncation=True,padding=True
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
        
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    
    #load pretrained t5 model
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    #data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    
    
    
    #generate train dataset
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator,
    )

    test_dataset = tokenized_datasets["test"].to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )

    generation_dataset = (
        tokenized_datasets["test"]
        .shuffle()
        .select(list(range(200)))
        .to_tf_dataset(
            batch_size=BATCH_SIZE,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )
    
    
    



    #metric function
    def metric_fn(eval_predictions):
        predictions, labels = eval_predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #     for label in labels:
    #         label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_l(decoded_labels, decoded_predictions)
        # We will print only the F1 score, you can use other aggregation metrics as well
        result = {"RougeL": result["f1_score"]}

        return result
    
    #model fit
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer)
    
    metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=generation_dataset, predict_with_generate=True
    )
    
    callbacks = [metric_callback]
    
    
    model.fit(
        train_dataset, validation_data=test_dataset, epochs=MAX_EPOCHS, callbacks=callbacks
    )
        
    model.save_pretrained('title_large')
    
    #generate summary

# =============================================================================
#     # map data correctly
#     def generate_summary(batch):
#         # Tokenizer will automatically set [BOS] <text> [EOS]
#         inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']
#         outputs = model.generate(input_ids, attention_mask=attention_mask)
#         # all special tokens including will be removed
#         output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# 
#         batch["pred"] = output_str
# 
#         return batch
# 
#     results = raw_datasets['test'].map(generate_summary, batched=True, batch_size=4, remove_columns=['text', 'subject', 'date'])
#     
#     
#     return results
# =============================================================================
        
# =============================================================================
# #evaluate model
# 
# def evaluate(result):
#     actual_titles = results['title']
#     model_output = results['pred']
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = list()
#     for output,actual in zip(model_output,actual_titles):
#         s = scorer.score(output,actual)
#         scores.append(s['rougeL'].fmeasure)
# 
#     print('Evaluation result',np.mean(scores))
#     return scores
#     
# =============================================================================
if __name__=="__main__":
    print('Running training script')
    train()
    #evaluate(results)
    