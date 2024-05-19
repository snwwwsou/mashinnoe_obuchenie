import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных и разделение на матрицу признаков и зависимую переменную
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print("Матрица признаков\n", X[:5])
print("Зависимая переменная\n", y[:5])

# Обработка пропущенных значений (если требуется)
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# Обработка категориальных данных (если требуется)
# Замена категории кодом (LabelEncoder)
# from sklearn.preprocessing import LabelEncoder
# labelencoder_y = LabelEncoder()
# print("Зависимая переменная до обработки")
# print(y)
# y = labelencoder_y.fit_transform(y)
# print("Зависимая переменная после обработки")
# print(y)

# Применение OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# print("Перекодировка категориального признака")
# print(X)

# Разделение выборки на тестовую и тренировочную

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=0)

# Обучение линейной модели регрессии

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Предсказание, обработка и визуализация результатов
y_pred = regressor.predict(X_test)
print(y_pred)

plt.figure(0)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.figure(1)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
