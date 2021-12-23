from flask import Flask, render_template, request, send_file
import pickle
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from transformers import T5TokenizerFast
from torch.utils.data import Dataset,DataLoader
from werkzeug.utils import secure_filename
import os
import csv
from rouge import Rouge
import graph
import time
from graph import make_graph
import spacy
from collections import Counter
from spacy import displacy
from pathlib import Path
import pandas
# import pyvips
from bleu import list_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

nlp = spacy.load('en_core_web_sm')

rouge = Rouge()

#model classes
class NewsSummaryDataset(Dataset):
    def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: T5TokenizerFast,
    text_max_token_len: int=512,
    summary_max_token_len: int=128
    ):
        self.tokenizer= tokenizer
        self.data=data
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data_row=self.data.iloc[index]
        
        text=data_row['document']
        
        text_encoding=tokenizer(
        text,
        max_length=self.text_max_token_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt')
        
        summary_encoding=tokenizer(
        data_row['summary'],
        max_length=self.summary_max_token_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt')
        
        labels=summary_encoding['input_ids']
        labels[labels==0]= -100
                 
        return dict(
        text=text,
        summary=data_row['summary'],
        text_input_ids=text_encoding['input_ids'].flatten(),
        text_attention_masks=text_encoding['attention_mask'].flatten(),
        labels=labels.flatten(),
        labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )
    
class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
    self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        batch_size:int=8,
        text_max_token_len:int=512,
        summary_max_token_len:int=128
    ):
        super().__init__()
        
        self.train_df=train_df
        self.test_df=test_df
        self.batch_size=batch_size
        self.tokenizer=tokenizer
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
        
    def setup(self,stage=None):
        self.train_dataset=NewsSummaryDataset(
        self.train_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
        )
        
        self.test_dataset=NewsSummaryDataset(
        self.test_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
        )
        
    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2)
        
    def val_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2)

    
model_name='t5-base'
tokenizer=T5TokenizerFast.from_pretrained(model_name)

class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        
    def forward(self,input_ids, attention_mask, decoder_attention_mask, labels=None):
        output=self.model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask)
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids=batch['text_input_ids']
        attention_mask=batch['text_attention_masks']
        labels=batch['labels']
        labels_attention_mask=batch['labels_attention_mask']
        
        loss, outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels)
        
        self.log('train_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids=batch['text_input_ids']
        attention_mask=batch['text_attention_masks']
        labels=batch['labels']
        labels_attention_mask=batch['labels_attention_mask']
        
        loss, outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels)
        
        self.log('val_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids=batch['text_input_ids']
        attention_mask=batch['text_attention_masks']
        labels=batch['labels']
        labels_attention_mask=batch['labels_attention_mask']
        
        loss, outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels)
        
        self.log('test_loss',loss,prog_bar=True,logger=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
           
     
T5_CNN = pickle.load(open('T5_CNN','rb'))

def summarize(text, length, model_name):
    text_encoding=tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors='pt')
    
    generated_ids= model_name.model.generate(
    input_ids=text_encoding['input_ids'],
    attention_mask=text_encoding['attention_mask'],
    max_length=length,
    num_beams=2,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True)
      
    preds=[
        tokenizer.decode(gen_id, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    
    return "".join(preds)

def rouge_out_test(hypothesis, reference):
    rouge_output = rouge.get_scores(hypothesis, reference)
    rouge_output = rouge_output[0]
    res = {key: rouge_output[key] for key in rouge_output.keys()
                               & {'rouge-l'}}
    dict_pairs = res.items()
    pairs_iterator = iter(dict_pairs)
    first_pair = next(pairs_iterator)   
    value = first_pair[1]
    value = {key: value[key] for key in value.keys()
                               & {'f'}}
    for elem in value.values():
        output = elem
    output = int(output * 100)
    return output
def bleu_cal_test(hypothesis, reference):
    # print(hypothesis)
    # print(reference)
    ref = [reference.split()]
    hypo = [hypothesis.split()]
    bscore = corpus_bleu([ref], hypo)
    bscore = round(bscore*100, 2)
    return bscore

def rouge_out(hypothesis, reference):
    total_rouge_score = 0
    for i in range(1000):
        rouge_output = rouge.get_scores(hypothesis[i],reference[i])
        rouge_output = rouge_output[0]
        res = {key: rouge_output[key] for key in rouge_output.keys()
                                    & {'rouge-l'}}
        dict_pairs = res.items()
        pairs_iterator = iter(dict_pairs)
        first_pair = next(pairs_iterator)   
        value = first_pair[1]
        value = {key: value[key] for key in value.keys()
                                    & {'f'}}
        for elem in value.values():
            output = elem
        total_rouge_score += output
    rscore = total_rouge_score / 1000
    rscore = round(rscore*100, 2)
    return rscore

def bleu_cal(hypothesis, reference):
    total_bleu_score = 0
    for i in range(1000):
        bleu_score = sentence_bleu([list(hypothesis[i].split(' '))], list(reference[i].split(' ')), weights=(1,0,0,0))
        total_bleu_score += bleu_score
    bscore = total_bleu_score / 1000
    bscore = round(bscore, 4)
    bscore = bscore * 100
    return bscore

def frequent_words(text):
    all_stopwords = nlp.Defaults.stop_words  
    sent = nlp(text)
    sent = [ent.text for ent in sent if ent.pos_ == 'NOUN']
    tokens_without_sw = [word for word in sent if not str(word).lower() in all_stopwords]
    test_list = Counter(tokens_without_sw).most_common(4)
    words = [lis[0] for lis in test_list]
    freq = [lis[1] for lis in test_list]
    return words, freq

def named_entity(text):
    sentence_nlp = nlp(text)
    svg = displacy.render(sentence_nlp, style='ent')
    time1 = time.time()
    new_graph_name1 = "entity" + str(time1) + ".svg"
    for filename in os.listdir('static/'):
        if filename.startswith('entity_'):  # not to remove other images
            os.remove('static/' + filename)
    output_path = Path('static/' + new_graph_name1)
    output_path.open("w", encoding="utf-8").write(svg)
    svg_code = open(output_path, 'rt').read()
    return svg_code

app = Flask(__name__)

app.config["CACHE_TYPE"] = "null"

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/summerizer')
def sum():
    return render_template('Abstractive.html')

basedir = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(basedir, 'static/results/sum.txt')
csv_path = os.path.join(basedir, 'static/results/data.csv')

@app.route("/result", methods = ['POST'])
def submit():
    if request.method == 'POST':
        file = request.files['file']
        words = request.form['number']
        if words == "":
            length = 150
        else:
            length = int(words)
        if file.filename == "":
            text = request.form['text']
            summary = summarize(text, length, T5_CNN)
            words, freq = frequent_words(text)
            new_graph_name = make_graph(text)
            with open(path, 'w') as f:
                f.write(str(summary))
            with open(csv_path, 'a') as f:
                f.write(",".join([text, summary]))
            time.sleep(2)
            return render_template('display.html', summary = summary, text= text, graph=new_graph_name, words=words, freq=freq)
        else:
            full_filename = secure_filename(file.filename)
            file.save(os.path.join('static',full_filename))
            with open(f"static/{full_filename}") as f :
                text = f.read()
            summary = summarize(text, length, T5_CNN) # Can we add variable of model name directly??
            words, freq = frequent_words(text)
            new_graph_name = make_graph(text)
            output_path = named_entity(summary)
            with open(path, 'w') as f:
                f.write(str(summary))
            with open(csv_path, 'a') as csvfile:
                fieldnames = ['text', 'summary']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # writer.writeheader()
                writer.writerow({'text': text, 'summary': summary})
            time.sleep(2)
            return render_template('display.html', summary = summary, text = text, graph=new_graph_name, graph_image=full_filename, words=words, freq=freq)

@app.route('/download', methods=['GET','POST'])
def download_file():
	return send_file(path, as_attachment=True)

@app.route('/score', methods=['POST'])
def score():
    if request.method == 'POST':
        file = request.files['file1']
        full_filename = secure_filename(file.filename)
        test_path = os.path.join(basedir,'static','results',full_filename)
        file.save(test_path)
        with open(test_path) as f:
            reference = f.read()
        with open(path) as f:
            summary = f.read()
        rscore = rouge_out_test(summary, reference)
        bscore = bleu_cal_test(summary, reference)
        return render_template('score.html', rscore= rscore, bscore = bscore)


@app.route('/tscore', methods=['POST'])
def tscore():
    if request.method == 'POST':
        file = request.files['file2']
        full_filename = secure_filename(file.filename)
        test_path = os.path.join(basedir,'static','eval',full_filename)
        file.save(test_path)
        df = pd.read_csv(test_path)
        reference = df['Predicted'].tolist()
        hypothesis = df['Original'].tolist()
        rscore1 = rouge_out(hypothesis, reference)
        bscore1 = bleu_cal(hypothesis, reference)
        return render_template('tscore.html', rscore1= rscore1, bscore1 = bscore1)

@app.route('/team')
def team():
	return render_template('team.html')


@app.route('/eval')
def eval():
    return render_template('eval.html')    


@app.route('/history')
def history():
    data = pandas.read_csv(csv_path) 
    myData = data.values 
    return render_template('history.html',  myData=myData) 


if __name__=='__main__':
	#app.run(host='192.168.134.147', port='8080', debug=True)
	app.run()
