from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Tải các mô hình đã lưu
model_pla = joblib.load('perceptron.pkl')
model_id3 = joblib.load('id3.pkl')
model_mlp = joblib.load('mlp.pkl')
model_bagging = joblib.load('bagging.pkl')

# Tải label encoder và scaler
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Định nghĩa ánh xạ từ số sang tên loài
species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Nhận dữ liệu từ form
            length = float(request.form['length'])
            depth = float(request.form['depth'])
            
            # Tạo mảng numpy cho dữ liệu đầu vào
            input_data = np.array([[length, depth]])
            
            # Chuẩn hóa dữ liệu đầu vào
            input_data_scaled = scaler.transform(input_data)
            
            # Dự đoán từ các mô hình
            predictions = {
                'PLA': int(model_pla.predict(input_data_scaled)[0]),
                'ID3': int(model_id3.predict(input_data_scaled)[0]),
                'MLP': int(model_mlp.predict(input_data_scaled)[0]),
                'Bagging': int(model_bagging.predict(input_data_scaled)[0])
            }
            
            # Chuyển đổi số nguyên thành tên loài
            predictions_named = {key: species_map[value] for key, value in predictions.items()}
            
            return render_template('result.html', predictions=predictions, predictions_named=predictions_named)
        except Exception as e:
            return f"Đã xảy ra lỗi: {str(e)}"
    return "No data submitted."

if __name__ == '__main__':
    app.run(debug=True)