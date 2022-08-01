import argparse
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MAX_LENGTH = 62


def train(data_path, model_path):
    # ==== load model
    model_name = "bvanaken/clinical-assertion-negation-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)


    # === load data
    dataset = load_dataset('csv', data_files=[data_path], column_names=["text", "label"], skiprows=1)


    # === load training and evaluation data
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        # tokenized["label"] = [int(label) for label in examples["label"]]
        return tokenized
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(364))
    small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))

    # === define the class to compute metrics
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

    # === define training arguments
    training_args = TrainingArguments(output_dir=model_path, evaluation_strategy="epoch")

    # === train
    trainer = Trainer(
      model=model,
      args=training_args, 
      train_dataset=small_train_dataset,
      eval_dataset=small_eval_dataset,
      compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(model_path)
    print("Fine tuning is successfully completed!")

def test(model_path):
    # === load fine tuned model
    model_name = "bvanaken/clinical-assertion-negation-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    
    # === prepare input and run model
    input = ["The patient has [entity] high fever [entity], denied any shortness of breath .",
             "The patient has high fever, denied any [entity] shortness of breath [entity] .",]
    output = classifier(input)
    print(output)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="Train or test a model",
    )
    FLAGS, unparsed = parser.parse_known_args()

    data_path = "../../data/datasets/assertion_detection/nursing_labels_mapped.csv"
    model_path = "models"

    if FLAGS.mode == "train":
        train(data_path, model_path)
    elif FLAGS.mode == "test":
        test(model_path)
    else:
        raise ValueError("Unknown mode {} detected. Only train or test are supported.")