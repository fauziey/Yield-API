from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all the models
def load_models():
    models = {}
    model_names = ['ct', 'best_dt', 'best_rf', 'best_xgb', 'best_final_xgb']
    
    for model_name in model_names:
        with open(f'{model_name}.pkl', 'rb') as file:
            models[model_name] = pickle.load(file)
    
    return models

# Load models when the app starts
loaded_models = load_models()

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    Year = float(request.form.get('Year'))
    Min_temp = float(request.form.get('Min_temp'))
    Max_temp = float(request.form.get('Max_temp'))
    AvgTemp = float(request.form.get('AvgTemp'))
    Precipitation = float(request.form.get('Precipitation'))
    Pesticides = float(request.form.get('Pesticides'))
    Item = request.form.get('Item')

    # Create a DataFrame with the input data
    features = pd.DataFrame([[Year, Min_temp, Max_temp, AvgTemp, Precipitation, Pesticides, Item]], 
                            columns=['Year', 'Min_temp', 'Max_temp', 'AvgTemp', 'Precipitation', 'Pesticides', 'Item'])
    
    # Transform the features using the loaded ColumnTransformer
    transform_features = loaded_models['ct'].transform(features)
    
    # Predict using the individual models
    pred_dt = loaded_models['best_dt'].predict(transform_features)
    pred_rf = loaded_models['best_rf'].predict(transform_features)
    pred_xgb = loaded_models['best_xgb'].predict(transform_features)
    
    # Combine the predictions (stacking)
    stacked_preds = np.column_stack((pred_dt, pred_rf, pred_xgb))
    
    # Predict the final yield using the final XGBoost model
    predicted_yield = loaded_models['best_final_xgb'].predict(stacked_preds)
    
    # Prepare the result
    result = {
        'Year': Year,
        'Min_temp': Min_temp,
        'Max_temp': Max_temp,
        'AvgTemp': AvgTemp,
        'Precipitation': Precipitation,
        'Pesticides': Pesticides,
        'Item': Item,
        'Predicted_Yield': float(predicted_yield[0])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)