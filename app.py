from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the Random Forest model
with open('model/randomForest1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    
    # Convert form data to list of values
    data = [
        float(form_data['satisfactoryLevel']),
        float(form_data['lastEvaluation']),
        int(form_data['numberOfProjects']),
        int(form_data['avgMonthlyHours']),
        int(form_data['timeSpentCompany']),
        int(form_data['workAccident']),
        int(form_data['promotionInLast5years']),
        int(form_data['salary'])
    ]
    
    # Convert data to numpy array and reshape for the model
    data = np.array([data])
    
    # Make prediction
    prediction = model.predict(data)
    
    # Return result
    result = 'At High Risk of Leaving' if prediction[0] == 1 else 'Not at High Risk of Leaving'
    return render_template('result.html', prediction_text=f'Employee is {result}')

if __name__ == '__main__':
    app.run(debug=True)
