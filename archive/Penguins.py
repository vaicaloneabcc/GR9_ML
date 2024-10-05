import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Tải dữ liệu
url = "penguins_lter.csv"
penguins = pd.read_csv(url)

# Khám phá dữ liệu
print(penguins.head())

# Tiền xử lý dữ liệu
# Xóa các hàng có giá trị thiếu
penguins.dropna(inplace=True)

# Mã hóa các biến phân loại
label_encoder = LabelEncoder()
penguins['species'] = label_encoder.fit_transform(penguins['species'])
penguins['sex'] = label_encoder.fit_transform(penguins['sex'])
penguins['island'] = label_encoder.fit_transform(penguins['island'])

# Chia dữ liệu thành features và label
X = penguins[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island']]
y = penguins['species']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện và đánh giá PLA
pla = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
pla.fit(X_train, y_train)
y_pred_pla = pla.predict(X_test)
pla_accuracy = accuracy_score(y_test, y_pred_pla)
print("PLA Accuracy:", pla_accuracy)

# Huấn luyện và đánh giá ID3
id3 = DecisionTreeClassifier(random_state=42)
id3.fit(X_train, y_train)
y_pred_id3 = id3.predict(X_test)
id3_accuracy = accuracy_score(y_test, y_pred_id3)
print("ID3 Accuracy:", id3_accuracy)

# Huấn luyện và đánh giá MLP
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print("MLP Accuracy:", mlp_accuracy)