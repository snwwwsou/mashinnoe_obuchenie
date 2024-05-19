import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Загрузка данных из файла "iris.data" и определение типа данных для каждого элемента
dt = np.dtype("f8, f8, f8, f8, U30")
data = np.genfromtxt("iris.data", delimiter=",", dtype=dt)

# Извлечение данных о длине и ширине чашелистика и лепестка из общего набора данных
sepal_length = [dot[0] for dot in data]
sepal_width = [dot[1] for dot in data]
petal_length = [dot[2] for dot in data]
petal_width = [dot[3] for dot in data]

# Построение графиков для различных комбинаций признаков

# График 1: Длина чашелистика vs. Ширина чашелистика
plt.figure(1)
setosa_1 = plt.plot(sepal_length[:50], sepal_width[:50], 'ro', label='Setosa')
versicolor_1 = plt.plot(sepal_length[50:100], sepal_width[50:100], 'g^', label='Versicolor')
virginica_1 = plt.plot(sepal_length[100:], sepal_width[100:], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.tight_layout()

# График 2: Длина чашелистика vs. Длина лепестка
plt.figure(2)
setosa_2 = plt.plot(sepal_length[:50], petal_length[:50], 'ro', label='Setosa')
versicolor_2 = plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label='Versicolor')
virginica_2 = plt.plot(sepal_length[100:], petal_length[100:], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.tight_layout()

# График 3: Ширина чашелистика vs. Ширина лепестка
plt.figure(3)
setosa_3 = plt.plot(sepal_width[:50], petal_width[:50], 'ro', label='Setosa')
versicolor_3 = plt.plot(sepal_width[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica_3 = plt.plot(sepal_width[100:], petal_width[100:], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.tight_layout()

# График 4: Длина лепестка vs. Ширина лепестка
plt.figure(4)
setosa_4 = plt.plot(petal_length[:50], petal_width[:50], 'ro', label='Setosa')
versicolor_4 = plt.plot(petal_length[50:100], petal_width[50:100], 'g^', label='Versicolor')
virginica_4 = plt.plot(petal_length[100:], petal_width[100:], 'bs', label='Virginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.tight_layout()

# Отображение графиков
plt.show()

# Создание списков для каждого вида ириса
setosa_sepal_length = sepal_length[:50]
setosa_sepal_width = sepal_width[:50]
setosa_petal_length = petal_length[:50]
setosa_petal_width = petal_width[:50]
versicolor_sepal_length = sepal_length[50:100]
versicolor_sepal_width = sepal_width[50:100]
versicolor_petal_length = petal_length[50:100]
versicolor_petal_width = petal_width[50:100]
virginica_sepal_length = sepal_length[100:]
virginica_sepal_width = sepal_width[100:]
virginica_petal_length = petal_length[100:]
virginica_petal_width = petal_width[100:]

# Создание таблицы с основными статистическими показателями
table = [["", "Max", "Min", "Mean", "Std"],
         ["Sepal Length", np.max(sepal_length), np.min(sepal_length), np.mean(sepal_length), np.std(sepal_length)],
         ["Sepal Width", np.max(sepal_width), np.min(sepal_width), np.mean(sepal_width), np.std(sepal_width)],
         ["Petal Length", np.max(petal_length), np.min(petal_length), np.mean(petal_length), np.std(petal_length)],
         ["Petal Width", np.max(petal_width), np.min(petal_width), np.mean(petal_width), np.std(petal_width)]]
table += [
    ["Setosa Sepal Length", np.max(setosa_sepal_length), np.min(setosa_sepal_length), np.mean(setosa_sepal_length),
     np.std(setosa_sepal_length)],
    ["Setosa Sepal Width", np.max(setosa_sepal_width), np.min(setosa_sepal_width), np.mean(setosa_sepal_width),
     np.std(setosa_sepal_width)],
    ["Setosa Petal Length", np.max(setosa_petal_length), np.min(setosa_petal_length), np.mean(setosa_petal_length),
     np.std(setosa_petal_length)],
    ["Setosa Petal Width", np.max(setosa_petal_width), np.min(setosa_petal_width), np.mean(setosa_petal_width),
     np.std(setosa_petal_width)]]
table += [
    ["Versicolor Sepal Length", np.max(versicolor_sepal_length), np.min(versicolor_sepal_length),
     np.mean(versicolor_sepal_length), np.std(versicolor_sepal_length)],
    ["Versicolor Sepal Width", np.max(versicolor_sepal_width), np.min(versicolor_sepal_width),
     np.mean(versicolor_sepal_width), np.std(versicolor_sepal_width)],
    ["Versicolor Petal Length", np.max(versicolor_petal_length), np.min(versicolor_petal_length),
     np.mean(versicolor_petal_length), np.std(versicolor_petal_length)],
    ["Versicolor Petal Width", np.max(versicolor_petal_width), np.min(versicolor_petal_width),
     np.mean(versicolor_petal_width), np.std(versicolor_petal_width)]]
table += [
    ["Virginica Sepal Length", np.max(virginica_sepal_length), np.min(virginica_sepal_length),
     np.mean(virginica_sepal_length), np.std(virginica_sepal_length)],
    ["Virginica Sepal Width", np.max(virginica_sepal_width), np.min(virginica_sepal_width),
     np.mean(virginica_sepal_width), np.std(virginica_sepal_width)],
    ["Virginica Petal Length", np.max(virginica_petal_length), np.min(virginica_petal_length),
     np.mean(virginica_petal_length), np.std(virginica_petal_length)],
    ["Virginica Petal Width", np.max(virginica_petal_width), np.min(virginica_petal_width),
     np.mean(virginica_petal_width), np.std(virginica_petal_width)]]
# Запись таблицы в файл "iris_statistics.txt"
with open('iris_statistics.txt', 'w') as f:
    f.write(tabulate(table, headers='firstrow', tablefmt='grid'))