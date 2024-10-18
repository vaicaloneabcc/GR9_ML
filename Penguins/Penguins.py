# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Đọc dữ liệu
url = "penguins_size.csv"  # Đường dẫn đến file dữ liệu
data = pd.read_csv(url, delimiter=',')

# Xóa các hàng có giá trị thiếu
data.dropna(inplace=True)

# Ánh xạ giá trị của species sang số nguyên
species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
data['species'] = data['species'].map(species_map)

# Chia dữ liệu thành features và label
X = data[['culmen_length_mm', 'culmen_depth_mm']]
y = data['species']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lưu scaler
joblib.dump(scaler, 'scaler.pkl')

# Huấn luyện và lưu mô hình PLA
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train_scaled, y_train)
joblib.dump(perceptron, 'perceptron.pkl')

# Dự đoán và đánh giá mô hình PLA
y_pred_pla = perceptron.predict(X_test_scaled)
print("Kết quả dự đoán của mô hình PLA trên tập kiểm tra:")
print("Độ chính xác:", accuracy_score(y_test, y_pred_pla))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred_pla))
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_pla))

# Huấn luyện và lưu mô hình ID3
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3.fit(X_train_scaled, y_train)
joblib.dump(id3, 'id3.pkl')

# Dự đoán và đánh giá mô hình ID3
y_pred_id3 = id3.predict(X_test_scaled)
print("\nKết quả dự đoán của mô hình ID3 trên tập kiểm tra:")
print("Độ chính xác:", accuracy_score(y_test, y_pred_id3))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred_id3))
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_id3))

# Huấn luyện và lưu mô hình MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
joblib.dump(mlp, 'mlp.pkl')

# Dự đoán và đánh giá mô hình MLP
y_pred_mlp = mlp.predict(X_test_scaled)
print("\nKết quả dự đoán của mô hình MLP trên tập kiểm tra:")
print("Độ chính xác:", accuracy_score(y_test, y_pred_mlp))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred_mlp))
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_mlp))

# Tạo, huấn luyện và lưu mô hình Bagging với Decision Tree
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'), n_estimators=10, random_state=42)
bagging_clf.fit(X_train_scaled, y_train)
joblib.dump(bagging_clf, 'bagging.pkl')

# Dự đoán trên tập kiểm tra
y_pred_bagging = bagging_clf.predict(X_test_scaled)

# In kết quả
print("\nKết quả dự đoán của Bagging:")
print("Độ chính xác:", accuracy_score(y_test, y_pred_bagging))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred_bagging))
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_bagging))

# Lưu label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y)
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nTất cả các mô hình và scaler đã được lưu.")

# Kiểm tra việc tải mô hình
print("\nKiểm tra tải mô hình:")
model_pla = joblib.load('perceptron.pkl')
model_id3 = joblib.load('id3.pkl')
model_mlp = joblib.load('mlp.pkl')
model_bagging = joblib.load('bagging.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

print("Tất cả các mô hình, scaler và label encoder đã được tải thành công.")

# Tạo một mẫu dữ liệu để kiểm tra
sample_data = np.array([[50,15.3]])  # Giả sử đây là dữ liệu mẫu
sample_data_scaled = loaded_scaler.transform(sample_data)

# Dự đoán với các mô hình đã tải
print("\nKết quả dự đoán với dữ liệu mẫu:")
print("PLA:", loaded_label_encoder.inverse_transform([model_pla.predict(sample_data_scaled)[0]])[0])
print("ID3:", loaded_label_encoder.inverse_transform([model_id3.predict(sample_data_scaled)[0]])[0])
print("MLP:", loaded_label_encoder.inverse_transform([model_mlp.predict(sample_data_scaled)[0]])[0])
print("Bagging:", loaded_label_encoder.inverse_transform([model_bagging.predict(sample_data_scaled)[0]])[0])