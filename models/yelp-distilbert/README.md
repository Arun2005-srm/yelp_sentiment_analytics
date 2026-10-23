---
library_name: transformers
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: my-yelp-sentiment-model-finetuned
  results: []
datasets:
- Yelp/yelp_review_full
language:
- en
base_model:
- distilbert/distilbert-base-uncased
---


# my-yelp-sentiment-model-finetuned

This model is a fine-tuned DistilBERT model for sentiment analysis on Yelp reviews. It was trained on the [Yelp/yelp-review-full](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset.

It achieves the following results on the evaluation set:
- Loss: 0.3645
- Accuracy: 0.8485
- F1: 0.8495

## Model description

This model is a fine-tuned version of DistilBERT, a smaller and faster variant of BERT, specifically adapted for sentiment analysis.


## Intended uses & limitations

 - Sentiment analysis of yelp reviews.
 - Analyzing customer feeback for businesses.
 - Helping businesses see the information they need in a timely manner to allow them to answer them more succintly.
 - This model IS NOT a replacement for humans and will still have issues with sarcasm, do not trust it 100% and please double check its findings.

## Training and evaluation data

The model was trained and evaluated on the [Yelp/yelp-review-full](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset. It uses the full 650,000 rows of training data and evaluates on the included test set.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7.637168097601297e-06
- train_batch_size: 32
- eval_batch_size: 64
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 4

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|
| 0.3669        | 1.0   | 20313 | 0.3651          | 0.8484   | 0.8436 |
| 0.3343        | 2.0   | 40626 | 0.3645          | 0.8485   | 0.8495 |
| 0.3133        | 3.0   | 60939 | 0.3689          | 0.8497   | 0.8481 |
| 0.2815        | 4.0   | 81252 | 0.3773          | 0.8506   | 0.8492 |


### Framework versions

- Transformers 4.48.3
- Pytorch 2.6.0+cu124
- Datasets 3.4.0
- Tokenizers 0.21.0

### Evaluation Results
| Metric                  | Value                  |
|-------------------------|------------------------|
| eval_loss               | 0.36449944972991943    |
| eval_accuracy           | 0.84846                |
| eval_f1                 | 0.8494926642060322     |
| eval_runtime            | 169.6017               |
| eval_samples_per_second | 294.808                |
| eval_steps_per_second   | 4.611                  |
| epoch                   | 4.0                    |

## Preprocessing

Tokenized using the default distilbert-base-uncased tokenizer

## Model Architecture

The model uses the DistilBERT base uncased architecture.

## Fine tuning approach

The model was fine tuned using the hugging face trainer api.

## Example Usage

```python
from transformers import pipeline
model_path = "FinchW/my-yelp-sentiment-model-finetuned"
tokenizer_path = "distilbert/distilbert-base-uncased"
id2label = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}
label2id = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2
}
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, id2label=id2label, label2id=label2id)

pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True) 
#pass single string
print("Single String:")
display(pipe("My food was disgusting!"))
#pass list of strings
print("List of Strings:")
display(pipe(["My food was amazing", "my food was sub-par", "gross"]))
```
