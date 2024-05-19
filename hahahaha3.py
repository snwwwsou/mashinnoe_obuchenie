import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Загрузка данных и первичный анализ
data_source = 'iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'answer']
data = pd.read_csv(data_source, names=column_names)

# Визуализация данных
sns.pairplot(data, hue='answer', markers=["o", "s", "D"])
plt.show()

# Обучение и оценка модели KNN
X = data.drop('answer', axis=1)
y = data['answer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Кросс-валидация для подбора оптимального значения K
k_list = list(range(1, 50))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Определение оптимального значения K
optimal_k = k_list[cv_scores.index(max(cv_scores))]
print('Optimal value of K:', optimal_k)

# Визуализация кривой ошибок
plt.plot(k_list, [1 - x for x in cv_scores])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Misclassification Error')
plt.title('KNN Misclassification Error')
plt.show()

# Визуализация решающих границ
dX = data.iloc[:, 0:4]
dy = data['answer']
plot_markers = ['r*', 'g^', 'bo']
answers = dy.unique()

# Создаем подграфики для каждой пары признаков
f, places = plt.subplots(4, 4, figsize=(16, 16))
fmin = dX.min() - 0.5
fmax = dX.max() + 0.5
plot_step = 0.05

# Обходим все subplot
for i in range(0, 4):
    for j in range(0, 4):
        # Строим решающие границы
        if i != j:
            # Создаем сетку значений для текущих признаков
            xx, yy = np.meshgrid(np.arange(fmin.iloc[i], fmax.iloc[i], plot_step),
                                 np.arange(fmin.iloc[j], fmax.iloc[j], plot_step))
            # Обучаем модель и предсказываем значения для сетки
            model = KNeighborsClassifier(n_neighbors=13)
            model.fit(dX.iloc[:, [i, j]].values, dy)
            p = model.predict(np.c_[xx.ravel(), yy.ravel()])
            p = p.reshape(xx.shape)
            p[p == answers[0]] = 0
            p[p == answers[1]] = 1
            p[p == answers[2]] = 2
            p = p.astype('int32')
            # Отображаем решающие границы на графике
            places[i, j].contourf(xx, yy, p, cmap='Pastel1')

        # Обход всех классов
        for idx, answer in enumerate(answers):
            if i == j:
                # Отображаем гистограмму распределения значений признака
                places[i, j].hist(dX.loc[dy == answer, dX.columns[i]],
                                  color=plot_markers[idx][0],
                                  histtype='step')
            else:
                # Отображаем точки классов на графике
                places[i, j].plot(dX.loc[dy == answer, dX.columns[i]], dX.loc[dy == answer, dX.columns[j]],
                                  plot_markers[idx],
                                  label=answer, markersize=6)
        # Добавляем подписи к осям
        if j == 0:
            places[i, j].set_ylabel(dX.columns[i])
        if i == 3:
            places[i, j].set_xlabel(dX.columns[j])

plt.show()