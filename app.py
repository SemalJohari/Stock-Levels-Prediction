from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

model = pickle.load(open("model_pickle.pkl", 'rb'))

app = Flask(__name__)

@app.route('/Analysis')
def analysis():
    return render_template("sales_report.html")

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        unit_price = float(request.form["unit_price"])
        quantity = request.form.get("quantity")
        if quantity is not None and quantity.isdigit():
            quantity = int(quantity)
        else:
            quantity = 0

        timestamp_day_of_month = int(request.form["timestamp_day_of_month"])
        timestamp_day_of_week = int(request.form["timestamp_day_of_week"])
        timestamp_hour = int(request.form["timestamp_hour"])

        features = pd.DataFrame(data=[[quantity, temperature, unit_price, timestamp_day_of_month, timestamp_day_of_week, 
                              timestamp_hour]], columns=['quantity', 'temperature', 'unit_price', 'timestamp_day_of_month',
                                                         'timestamp_day_of_week', 'timestamp_hour'])

        scaled_numerical_features = scaler.fit_transform(features)
        scaled_numerical_features_df = pd.DataFrame(scaled_numerical_features, columns=['quantity', 'temperature', 
                                                                                        'unit_price', 
                                                                                        'timestamp_day_of_month', 
                                                                                        'timestamp_day_of_week', 
                                                                                        'timestamp_hour'])

        predictions = model.predict(scaled_numerical_features_df)[0]
        print(predictions)
        return render_template("index.html", prediction_text="Predicted Stock Levels: {}".format(predictions))
    
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)