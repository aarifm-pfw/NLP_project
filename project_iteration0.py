#Import the necessary libraries
from transformers import pipeline
from tqdm.notebook import tqdm
from datasets import load_dataset
import numpy as np
import math
from sklearn.metrics import classification_report

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#Test case 1 using sample input text
candidate_labels = ["world", "sports", "business", "sci/tech"]
text1 = "Veteran inventor in market float Trevor Baylis, the veteran inventor famous for creating the Freeplay clockwork radio, is planning to float his company on the stock market."
result = classifier(text1, candidate_labels)
print(result)

#Test case 2 using sample input text
text2 = "This Date in Baseball - Aug. 17 (AP) AP - 1904  #151; Jesse Tannehill of the Boston Red Sox pitched a no-hitter, beating the Chicago White Sox 6-0."
result1 = classifier(text2, candidate_labels)
print(result1)

#Evaluating using ag_news dataset
dataset = load_dataset('ag_news')
candidate_labels = ["world", "sports", "business", "sci/tech"]
predictions = []
for offset in tqdm(range(math.ceil(len(dataset["test"])/16))):
    preds = classifier([dataset["test"][16*offset+i]["text"] for i in range(16) if 16*offset+i<len(dataset["test"])], candidate_labels)
    pred_labels = [pred["labels"][np.argmax(pred["scores"])] for pred in preds]
    predictions.extend([candidate_labels.index(pred_label) for pred_label in pred_labels])
#Printing the report
print(classification_report([x["label"] for x in dataset["test"]], predictions))