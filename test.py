import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from datasets import Dataset
from transformers import TFAutoModelForSeq2SeqLM,AutoTokenizer

#change path according to your need
# I have provided separate files for tokenizer and pretrained models
submission = pd.read_csv('test.csv')
test_data = Dataset.from_pandas(submission)
tokenizer = AutoTokenizer.from_pretrained("tokenizer_title")
model= TFAutoModelForSeq2SeqLM.from_pretrained("title_large")

def predict(texts):
    # write code to output a list of title for each text input to the predict method
    def generate_summary(batch):
        inputs = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        batch["pred"] = output_str
        
        return batch
    results = test_data.map(generate_summary, batched=True, batch_size=4, remove_columns=['text', 'subject', 'date'])
    return results['pred']


def test_model():
    pred = predict(test_data)
    submission['predicted_title'] = pred
    submission.to_csv('test.csv',index=False)


def evaluate(model_output,actual_titles):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = list()
    for output,actual in zip(model_output,actual_titles):
        s = scorer.score(output,actual)
        scores.append(s['rouge1'].fmeasure)

    print('Evaluation result',np.mean(scores))
    return scores




if __name__=="__main__":
    #write model loading code here

    test_model()
    evaluate(submission['predicted_title'].values, submission['title'].values)
