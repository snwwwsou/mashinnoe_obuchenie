import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import graphviz
from sklearn.metrics import accuracy_score


# Загрузка данных
data_source = 'iris.data'
d = pd.read_csv(data_source, header=None,
                names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'answer'])

# Подготовка данных
dX = d.iloc[:, 0:4]
dy = d['answer']

# Разделение на обучающую и тестовую выборки
X_train, X_holdout, y_train, y_holdout = train_test_split(dX, dy, test_size=0.3, random_state=12)

# Обучение модели
tree = DecisionTreeClassifier(max_depth=5, random_state=21, max_features=2)
tree.fit(X_train, y_train)

# Оценка модели на тестовой выборке
tree_pred = tree.predict(X_holdout)
accuracy = accuracy_score(y_holdout, tree_pred)
print('Accuracy:', accuracy)

# Выбор оптимальной глубины дерева с помощью кросс-валидации
d_list = list(range(1, 20))
cv_scores = []
for d in d_list:
    tree = DecisionTreeClassifier(max_depth=d, random_state=21, max_features=2)
    scores = cross_val_score(tree, dX, dy, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Определение минимальной ошибки
MSE = [1 - x for x in cv_scores]
d_min = min(MSE)
all_d_min = [d_list[i] for i, mse in enumerate(MSE) if mse <= d_min]
print('Оптимальные значения max_depth:', all_d_min)

# Строим график
plt.plot(d_list, MSE)
plt.xlabel('Макс. глубина дерева (max_depth)')
plt.ylabel('Ошибка классификации (MSE)')
plt.show()

# Поиск оптимальных параметров с помощью GridSearchCV
dtc = DecisionTreeClassifier(random_state=21, max_features=2)
tree_params = {'max_depth': range(1, 20), 'max_features': range(1, 4)}
tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
tree_grid.fit(dX, dy)

print('Лучшее сочетание параметров:', tree_grid.best_params_)
print('Лучшие баллы cross validation:', tree_grid.best_score_)

dot_data = export_graphviz(tree_grid.best_estimator_,
                           feature_names=dX.columns,
                           class_names=dy.unique(),
                           out_file=None,
                           filled=True, rounded=True)

graph = graphviz.Source(dot_data)
dtc = DecisionTreeClassifier(max_depth=5,
                             random_state=21,
                             max_features=1)
# Обучаем
dtc.fit(dX.values, dy)
# Предсказываем
res = dtc.predict([[5.1, 3.5, 1.4, 0.2]])
dot_data = export_graphviz(dtc,
                           feature_names=dX.columns,
                           class_names=dy.unique(),
                           out_file=None,
                           filled=True, rounded=True)
graphviz.Source(dot_data)
# Визуализация данных и решающих границ
plot_markers = ['r*', 'g^', 'bo']
answers = dy.unique()
labels = dX.columns.values
f, places = plt.subplots(4, 4, figsize=(16, 16))
fmin = dX.min().values - 0.5
fmax = dX.max().values + 0.5
plot_step = 0.02

# Обходим все subplot
for i in range(0, 4):
    for j in range(0, 4):
        # Строим решающие границы
        if i != j:
            xx, yy = np.meshgrid(np.arange(fmin[i], fmax[i], plot_step, dtype=float),
                                 np.arange(fmin[j], fmax[j], plot_step, dtype=float))
            model = DecisionTreeClassifier(max_depth=2, random_state=21, max_features=3)
            model.fit(dX.iloc[:, [i, j]].values, dy.values)
            p = model.predict(np.c_[xx.ravel(), yy.ravel()])
            p = p.reshape(xx.shape)
            p[p == answers[0]] = 0
            p[p == answers[1]] = 1
            p[p == answers[2]] = 2
            p = p.astype('int32')
            places[i, j].contourf(xx, yy, p, cmap=plt.cm.Pastel2)

        # Обход всех классов (Вывод обучающей выборки)
        for id_answer in range(len(answers)):
            idx = np.where(dy == answers[id_answer])
            if i == j:
                places[i, j].hist(dX.iloc[idx].iloc[:, i],
                                  color=plot_markers[id_answer][0],
                                  histtype='step')
            else:
                places[i, j].plot(dX.iloc[idx].iloc[:, i], dX.iloc[idx].iloc[:, j],
                                  plot_markers[id_answer],
                                  label=answers[id_answer], markersize=6)

        # Печать названия осей
        if j == 0:
            places[i, j].set_ylabel(labels[i])

        if i == 3:
            places[i, j].set_xlabel(labels[j])
plt.show()
