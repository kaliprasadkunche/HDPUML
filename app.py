# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def roorhome():
	return render_template('NewLogin.html')
	

@app.route('/Signup')
def Signup():
	return render_template('SignUp.html')
	
@app.route('/Signupdb', methods=['POST'])
def do_userregisterdb():
    return render_template('NewLogin.html')
	
	
@app.route('/LoginDB', methods=['POST'])
def do_login():
    return render_template('main.html')

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        

        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        import numpy as np
        from matplotlib import pyplot as plt
        
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import tree
        from sklearn.svm import SVC
        import pickle

        data1=pd.read_csv("heart.csv")
        data1.head()


        data1.shape


        data1.info()




        data1.isnull().sum()



        data1.describe()


        # # **4. EXPLORATORY DATA ANALYSIS**


        data1['target'].value_counts()

        data1['target'].value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8),shadow=True)



        data1.groupby(['sex'])['target'].value_counts()



        data1.groupby(['fbs'])['target'].value_counts()

        data1.groupby(['exang'])['target'].value_counts()




        data1.groupby(['slope'])['target'].value_counts()


        data1.groupby(['ca'])['target'].value_counts()



        data1.groupby(['thal'])['target'].value_counts()


        X = data1.drop(['target'],axis='columns')
        X.head(10)


        # In[34]:


        y = data1.target
        y.head(3)


        # In[35]:


        len(y)


        # In[36]:


        len(X)


        # ### Creating Training and Test sample variables.

        # In[37]:


        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


        # # **7. MODEL BUILDING**

        # ## 1. Naive Bayes classifier

        # In[38]:


        model_1 = MultinomialNB()
        model_1.fit(X_train, y_train)
        nb=model_1.score(X_test, y_test)
        nb


        # ## 2. Logistic Regression Algorithm.

        # In[39]:


        model_2 = LogisticRegression()
        model_2.fit(X_train, y_train)
        lr=model_2.score(X_test, y_test)
        lr


        # ## 3. Random Forest Algorithm

        # In[40]:


        model_3 = RandomForestClassifier(n_estimators=30)
        model_3.fit(X_train, y_train)
        rf=model_3.score(X_test, y_test)
        rf


        # ## 4. Decision Tree Algorithm

        # In[41]:


        model_4 = tree.DecisionTreeClassifier(criterion='entropy')
        model_4.fit(X_train, y_train)
        dt=model_4.score(X_train, y_train)
        dt


        # ## 5. Support Vector Machine

        # In[42]:


        model_5 = SVC()
        model_5.fit(X_train, y_train)
        sv=model_5.score(X_test, y_test)
        sv


        # # **8. MODEL COMPARISON**

        # ## Comparing MAchine Learning Models using their scores.
        # 

        # In[43]:


        accuracy = [nb,lr,rf,dt,sv]
        all_models = ['NaiveBayesClassifier','LogisticRegression','RandomForestClassifier','DecisonTreeClassifier','SVC']

        score_df = pd.DataFrame({'Algorithms': all_models, 'Accuracy_Score': accuracy})
        score_df.style.background_gradient(cmap="YlGnBu",high=1,axis=0)


        #model_4.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])


        x=model_4.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        print(x)

        #data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        #my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=x[0])
        
        

if __name__ == '__main__':
	app.run(debug=True)

