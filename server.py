from flask import Flask, request, jsonify
import tensorflow_hub as hub
import tensorflow_text
import random
from joblib import  load
import pandas as pd 

app = Flask(__name__)

# load classifier
print("Loading the model...")
clf = load('output/model.joblib') 
print("Model loaded. \n")

print("Loading original questions...")
labeled_qs = pd.read_csv('output/clusters.csv')
klusters = labeled_qs['label'].max()
clusters_examples = []
print("# of clusters: {}".format(klusters))
# we don't need all the examples, a couple for each cluster are enough. 
for k in range(klusters+1):
	all_cluster = list(labeled_qs[labeled_qs['label']==k]['question'])
	max_elems = max(len(all_cluster),100)
	clusters_examples.append(random.sample(all_cluster,max_elems))

print("Questions loaded.\n")

print("Loading original embedding model...")
#load embedding model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
print("Embedding loaded.\n")

@app.route('/predict',methods=['POST'])
def make_prediction():
	print(request)
	try:
		# Get the question (maybe not as json)
		question = request.form['question']

	except Exception as e:
		return(bad_request())

	if question=='':
		return(bad_request())

	print(question)

	# get vectors for the question
	q_emb = embed([question])
	
	# preprocess the data before using the model on it
	pred = clf.predict(q_emb)[0]

	# Get other questions in the same cluster
	same_cluster = random.sample(clusters_examples[pred],10)

	#Send the response
	responses = jsonify(sentences=same_cluster)
	responses.status_code = 200
	return responses


@app.route('/my400')
def bad_request():
    code = 400
    msg = 'Error'
    return msg, code

if __name__ == '__main__':
	app.run()