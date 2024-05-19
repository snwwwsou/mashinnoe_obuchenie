import numpy as np
import pandas as pd

# Загрузка данных и разделение на матрицу признаков и зависимую переменную
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print("Матрица признаков\n", X[:5])
print("Зависимая переменная\n", y[:5])


# Применение OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Определение столбцов, которые нужно перекодировать в бинарные признаки
columns_to_encode = [3]  # Предполагаем, что столбец 3 нужно перекодировать

# Создание объекта ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), columns_to_encode)  # Применение OneHotEncoder к столбцу 3
    ],
    remainder='passthrough'  # Оставление остальных столбцов без изменений
)

# Применение преобразований к данным
X = ct.fit_transform(X)
print("Перекодировка категориального признака")
# Вывод первых 4 строк полученной матрицы
print(X[:4, :])

# Для предотвращения мультиколлениврности необходимо избавиться от одной из фиктивных переменных, добавленных в результате обработки категориальных признаков
X = X[:, 1:]
print(X[:4, :])

# Разделение выборки на тестовую и тренировочную
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение линейной модели регрессии
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Обработка результатов, тюнинг модели
# Предсказание
y_pred = regressor.predict(X_test)
print(y_pred)

# Оптимизация модели
import statsmodels.api as sm

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
