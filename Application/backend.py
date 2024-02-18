import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def load_model():
    with open('decision', 'rb') as f:
        mod = pickle.load(f)
    return mod



model = load_model()

@app.route("/predict")
@cross_origin()
def home():
    dep = request.args.get('depen')
    edu = request.args.get('education')
    emp = request.args.get('employment')
    income = request.args.get('income')
    loan = request.args.get('loan')
    term = request.args.get('term')
    cibil = request.args.get('cibil')

    val = [[dep, edu, emp, income, loan, term, cibil]]
    col = ["no_of_dependents", 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score']

    df = pd.DataFrame(val, columns=col)
    model = load_model()

    pred = model.predict(df)

    if pred[0]==[0]:
        return 'rejected'
    else:
        return 'approved'


if __name__=="__main__":
    app.run(debug=True)