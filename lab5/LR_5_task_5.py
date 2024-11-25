import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

def remove_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.strftime('%H:%M')

def day_of_week(date_str):
    obj = datetime.strptime(date_str, '%m/%d/%y')
    return obj.strftime('%A')

def is_win(match_res):
    return 'yes' if 'W' in match_res else 'no'

input_file = 'Dodgers.events'
data = []

# Читання даних з файлу
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

# Перетворення у NumPy масив і вибір потрібних стовпців
data = np.array(data)
data = data[:, [0, 1, 4, 5, 3]]  # День тижня, час, команда противника, чи виграш, кількість машин

# Зведення даних у потрібний вигляд
data[:, 0] = np.vectorize(day_of_week)(data[:, 0])
data[:, 1] = np.vectorize(remove_seconds)(data[:, 1])
data[:, 3] = np.vectorize(is_win)(data[:, 3])

# Кодування ознак
label_encoder = []
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoder.append(le)

# Поділ на X та y
X = X_encoded[:, :-1].astype(int)  # Всі стовпці, крім останнього (кількість машин)
y = X_encoded[:, -1].astype(int)  # Останній стовпець

# Розбивка на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Налаштування та тренування моделі
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Тестова точка
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']  # Нові значення для прогнозування
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0

# Кодуємо тестову точку, якщо вона вже була в тренувальних даних
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(item)
    else:
        if item in label_encoder[count].classes_:
            test_datapoint_encoded[i] = int(label_encoder[count].transform([item])[0])
        else:
            test_datapoint_encoded[i] = -1
        count += 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозування
predicted_traffic = int(regressor.predict([test_datapoint_encoded])[0])
print("Predicted traffic:", predicted_traffic)
