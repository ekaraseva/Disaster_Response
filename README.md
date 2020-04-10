# Disaster_Response
This project is a part of the Udacity's Data Scientist Nanodegree Program.


## 1. Project Descriptions
The project analyse disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set containing real messages that were sent during disaster events. The machine learning pipeline was built to categorize these events so that the messages can be appointed to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 

The project devided into three parts: 
- ELT - extract, transform, clean and load data,
- Machine Learning Pipeline - create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model,
- FLask APP - display the results in a Flask web app

## 2. Installation

Clone or download the repository.

**Prerequisites**
- pandas
- numpy
- nlkt
- re
- pickle
- sklearn
- sqlalchemy
- sqlite3

For plotting:
- matplotlib
- seaborn

## 3. How to interact with the project
The following Python scripts should be able to run data preparation and model selection steps:

- python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
- python train_classifier.py ../data/DisasterResponse.db classifier.pkl

To run the web app the following steps shall be executed:

- go to app folder 'cd app'
- python run.py
- open new ternimal and execute env|grep WORK, you will sea SPACEID and SPACEDOMAIN address
- open browser and type https://SPACEID-3001.SPACEDOMAIN (SPACEID and SPACEDOMAIN from step above)
