import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Загрузка данных и разделение на матрицу признаков и зависимую переменную
dataset = pd.read_csv('Data5.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print("Матрица признаков\n", X)
print("Зависимая переменная\n", y)

# Обработка пропущенных значений
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X_without_nan = X.copy()
X_without_nan[:, 1:3] = imputer.transform(X[:, 1:3])
print(X_without_nan)
# Обработка категориальных данных

# Замена категории кодом (LabelEncoder)
labelencoder_y = LabelEncoder()
print("Зависимая переменная до обработки")
print(y)
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после обработки")
print(y)

# Применение OneHotEncoder
# Создаем список трансформеров
transformers = [
    ('onehot', OneHotEncoder(), [0]),
    ('imp', SimpleImputer(), [1, 2])
]
# Создаем копию "грязного" объекта: спропусками и некодированными категориями
X_dirty = X.copy()
# Создаем объект ColumnTransformer и передаем ему список трансформеров
ct = ColumnTransformer(transformers)
# Выполняем трансформацию признаков
X_transformed = ct.fit_transform(X_dirty)
print(X_transformed.shape)
print(X_transformed)

# Преобразование полученного многомерного массива обратно в Dataframe
X_data = pd.DataFrame(
    X_transformed,
    columns=['C1', 'C2', 'C3', 'Age', 'Salary'])
print(X_data)
