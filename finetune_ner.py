# Domain-Specific BERT NER Training


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np, evaluate, os

MODEL_REF = {
  "Medical": "emilyalsentzer/Bio_ClinicalBERT",
  "Legal": "nlpaueb/legal-bert-base-uncased",
  "General": "bert-base-multilingual-cased"
}

def main(domain):
    assert domain in MODEL_REF
    ds = load_dataset("conll2003", data_files={"train": f"{domain.lower()}/train.conll", "validation": f"{domain.lower()}/valid.conll"})
    label_list = ds["train"].features["ner_tags"].feature.names
    model = AutoModelForTokenClassification.from_pretrained(MODEL_REF[domain], num_labels=len(label_list))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REF[domain])
    def tokenize_align(batch):
        toks = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
        new_labels = []
        for i, labs in enumerate(batch["ner_tags"]):
            word_ids = toks.word_ids(batch_index=i)
            lab = [-100 if idx is None else labs[idx] for idx in word_ids]
            new_labels.append(lab)
        toks["labels"] = new_labels
        return toks
    tokenized = ds.map(tokenize_align, batched=True)
    args = TrainingArguments(output_dir=f"{domain.lower()}_ner", per_device_train_batch_size=8, per_device_eval_batch_size=8,
                              num_train_epochs=5, evaluation_strategy="epoch", save_strategy="epoch", weight_decay=0.01,
                              load_best_model_at_end=True)
    metric = evaluate.load("seqeval")
    def compute(p):
        preds = np.argmax(p.predictions, axis=-1)
        refs = p.label_ids
        true_preds = [[label_list[p] for (p,l) in zip(pre, lab) if l!=-100] for pre,lab in zip(preds, refs)]
        true_refs = [[label_list[l] for (p,l) in zip(pre, lab) if l!=-100] for pre,lab in zip(preds, refs)]
        res = metric.compute(predictions=true_preds, references=true_refs)
        return {k: res[f"overall_{k}"] for k in ["precision","recall","f1","accuracy"]}
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                      tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=compute)
    trainer.train()
    model.save_pretrained(f"bert_models/{domain}")
    tokenizer.save_pretrained(f"bert_models/{domain}")

if __name__=="__main__":
    for d in ["General","Medical","Legal"]:
        print(f"Training {d} model...")
        main(d)
