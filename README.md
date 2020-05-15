# Disaster Response Pipeline Project

### Installations
* numpy
* pandas
* sqlalchemy
* sqlite3
* joblib
* re
* nltk
* sklearn
* xgboost
* json
* plotly
* flask

### File Descriptions
data
* disaster_categories.csv : category data to process
* disaster_messages.csv : message data to process
* process_data.py : ETL pipeline script
* DisasterResponse.db : database to save clean data to

models
* train_classifier.py : ML pipeline script
* classifier.pkl : saved model 

app
* templates
  * go.html : classification result page of web app
  * master.html : main page of web app
  
* run.py : Flask file that runs app

ETL Pipeline Preparation.ipynb: ETL Pipeline Preparation Jupyter notebook

ML Pipeline Preparation.ipynb: ML Pipeline Preparation Jupyter notebook


LICENSE.txt: MIT License

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
