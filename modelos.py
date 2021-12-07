from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import csv
import urllib.request

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def return_models_labels(task):
	task = task
	partes = task.split('-')
	MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
	tokenizer = AutoTokenizer.from_pretrained(MODEL)
	model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
	if len(partes) > 1:
		mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/stance/mapping.txt"
	else:
		mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
	with urllib.request.urlopen(mapping_link) as f:
		html = f.read().decode('utf-8').split("\n")
		csvreader = csv.reader(html, delimiter='\t')
	labels = [row[1] for row in csvreader if len(row) > 1]
	return tokenizer, model, labels