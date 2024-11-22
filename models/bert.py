from transformers import pipeline
import tqdm

import torch


def get_bert_sentiment(texts):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pipeline("sentiment-analysis", model='autoevaluate/binary-classification', device=device)
        
    predictions = []

    for text in tqdm.tqdm(texts):
        
        pred = model(text)[0]['label']

        pred_bin = 0 if pred=='negative' else 1         

        predictions.append(pred_bin)

    return predictions




if __name__ == "__main__":


    texts = ['This was good',
             'This was bad',
             'This was ok']

    predictions = get_bert_sentiment(texts)

    print(predictions)