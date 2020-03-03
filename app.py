from flask import Flask,render_template,url_for,request
import pickle

from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
from fastai.vision import *

app = Flask(__name__)
path=Path('./model')
path1=Path('./static/img')
model = load_learner(path)
import pandas as pd
train=pd.read_csv('HeartDiseaseClassification.csv',sep=';')
target=train['target']
train.drop(['target'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,target, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clf.fit(X_train,y_train)



photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

# load the model from disk
#filename = 'nlp_model.pkl'
#clf = pickle.load(open(filename, 'rb'))
#cv=pickle.load(open('tranform.pkl','rb'))


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        img=open_image(path1/filename)
        pred_class,pred_idx,outputs = model.predict(img)
        return str(pred_class)
        
    return render_template('upload.html')	




@app.route('/heart', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
    	return (render_template('heart.html'))


    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp =  request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']

        fbs = request.form['fbs']
        restecg = request.form['restecg']
        car = request.form['car']
        thalach = request.form['thalach']
        exang = request.form['exang']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']

        input_variables = pd.DataFrame([[age, sex, cp, trestbps,chol, fbs, restecg, car, thalach, exang, slope,ca,thal]],
                                       columns=['age', 'sex', 'cp', 'trestbps','chol', 'fbs',
                                                'restecg', 'car', 'thalach', 'exang', 'slope','ca','thal'],
                                       dtype='float',
                                       index=['input'])

        predictions = clf.predict(input_variables)[0]
        print(predictions)

        return render_template('heart.html', original_input={'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,'chol':chol, 'fbs': fbs, 'restecg': restecg, 'car': car, 'thalach': thalach, 'exang': exang, 'slope': slope,'ca': ca, 'thal':thal},
                                     result=predictions)

#@app.route('/predict',methods=['POST'])
#def predict():

	#if request.method == 'POST':
		#message = request.form['message']
		#data = [message]
		#vect = cv.transform(data).toarray()
		#my_prediction = clf.predict(vect)
	#return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
