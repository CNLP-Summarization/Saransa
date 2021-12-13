from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import pickle
import image_cap
import time
import met
from openpyxl import load_workbook
import nltk
import string
from string import punctuation
punctuation = punctuation +'\n'
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from heapq import nlargest
import nltk_fi
import page_rank
import subprocess
from flask import send_from_directory
import t5
import gpt2
import interpolate
UPLOAD_FOLDER = '..'
# import pdfkit
incorrect_answers=[]
incorrect_answers2=[]
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.add_extension('jinja2.ext.loopcontrols')

@app.route('/')
def upload_file():
	return render_template('index.html')

@app.route('/abstractive_summ')
def Abstractive_summarization():
	return render_template('Abstractive.html')

@app.route('/display12', methods = ['GET', 'POST'])
def holla():
	if request.method == 'POST':
	       # text_data=request.form['text_data']
		summarizer=request.form['analyzer']
		      # summary_size=request.form['summary_size']
		f = request.files['file-csv']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],'utk.txt'))
		global incorrect_answers,incorrect_answers2
		if(summarizer=='enc_dec'):
			subprocess.Popen('onmt_translate -model ../trained_model.pt -src ../utk.txt -output ../res.txt',shell=True)
			time.sleep(30)
			
			with open("../res.txt", "r") as file:
				incorrect_answers = file.readlines()
			with open("../utk.txt", "r") as file:
				incorrect_answers2= file.readlines()
			return render_template('display.html',incorrect_answers=incorrect_answers,incorrect_answers2=incorrect_answers2,len=len(incorrect_answers))
		else:
			text_data=request.form['text_data']
			flag=1
			if(text_data==''):
				flag=0
			res=t5.abc(text_data,flag)
			
			with open("../res.txt", "r") as file:
				incorrect_answers = file.readlines()
			with open("../utk.txt", "r") as file:
				
				incorrect_answers2= file.readlines()
			return render_template('display.html',incorrect_answers=incorrect_answers,incorrect_answers2=incorrect_answers2,len=len(incorrect_answers))
		 
	  

@app.route('/dl2')
def process():
	return send_from_directory('..','res.txt', as_attachment=True,cache_timeout=0)

@app.route('/dl3')
def proce():
	return send_from_directory('.','export_dataframe.xlsx', as_attachment=True,cache_timeout=0)

@app.route('/perl_processing', methods = ['GET', 'POST'])



def pro():
	if request.method == 'POST':
		global incorrect_answers,incorrect_answers2
		      # text_data=request.form['text_data']
		      # summarizer=request.form['analyzer']
		      # summary_size=request.form['summary_size']
		f=request.files['file-csv']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],'actual_output.txt'))
		eval=request.form['evaluation']
		if eval=='bleu':
			subprocess.Popen('perl ../multi-bleu.perl ../actual_output.txt < ../res.txt > ../eval_score.txt',shell=True)
		elif eval=='human_evaluation':
			return render_template('human.html',incorrect_answers=incorrect_answers,incorrect_answers2=incorrect_answers2,len=len(incorrect_answers))
		elif eval=='rouge':
			subprocess.Popen('files2rouge ../actual_output.txt ../res.txt --ignore_empty > ../eval_score.txt',shell=True)
		else:
			met.abc()

		time.sleep(30)
		return send_from_directory('..','eval_score.txt', as_attachment=True,cache_timeout=0)


@app.route('/team')
def team():
	return render_template('team.html')

@app.route('/Analyser_result', methods = ['GET', 'POST'])
def analysis_result():
	if request.method == 'POST':
	     # text_data=request.form['text_data']
	     # summarizer=request.form['analyzer']
	     # summary_size=request.form['summary_size']
		f=request.files['actual']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],'actual_output.txt'))
		f=request.files['pred']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'],'res.txt'))
		eval=request.form['evaluation']
		if eval=='bleu':
			subprocess.Popen('perl ../multi-bleu.perl ../actual_output.txt < ../res.txt > ../eval_score.txt',shell=True)
		elif eval=='rouge':
			subprocess.Popen('files2rouge ../actual_output.txt ../res.txt --ignore_empty > ../eval_score.txt',shell=True)
		else:
			met.abc()
		time.sleep(30)
		return send_from_directory('..','eval_score.txt', as_attachment=True,cache_timeout=0)

@app.route('/excel',methods = ['GET', 'POST'])
def excel_convert():
	global incorrect_answers,incorrect_answers2
	if request.method == 'POST':
		arr=[]
		for i in range(len(incorrect_answers)):
			arr.append(request.form[str(i)])
		
		score_human=interpolate.funct(arr)
		incorrect_answers.append('interpolated score')
		arr.append(score_human)
		incorrect_answers2.append(' ')
		data_fra={ 'original_text': incorrect_answers2,
			'predicted_summary': incorrect_answers,
			'human_evaluation': arr
			}

		df = pd.DataFrame(data_fra, columns = ['original_text','predicted_summary','human_evaluation'])
		df.to_excel (r'export_dataframe.xlsx', index = False, header=True)
		return send_from_directory('.','export_dataframe.xlsx', as_attachment=True,cache_timeout=0)



if __name__=='__main__':
	app.run(host='192.168.132.69',port='8080')
