import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "telecom_churn.csv"
data = pd.read_csv(data_path)

# Гистограмма количества звонков в службу поддержки
plt.figure(figsize=(8, 6))
data['Customer service calls'].hist()
plt.title('Гистограмма количества звонков в службу поддержки')
plt.xlabel('Количество звонков')
plt.ylabel('Частота')
plt.show()
# Boxplot для общего времени разговора в дневное время
plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Total day minutes'])
plt.title('Boxplot для общего времени разговора в дневное время')
plt.ylabel('Общее время разговора (минуты)')
plt.show()

# Boxplot для общего времени разговора в дневное время по топ-3 штатам с наибольшим временем разговора
top_states = data.groupby('State')['Total day minutes'].sum().nlargest(3).index
plt.figure(figsize=(10, 6))
sns.boxplot(x='Total day minutes', y='State', data=data[data['State'].isin(top_states)], hue='State', palette='Set2', legend=False)
plt.title('Boxplot для общего времени разговора в дневное время по топ-3 штатам')
plt.xlabel('Общее время разговора (минуты)')
plt.ylabel('Штат')
plt.show()

# Гистограмма количества абонентов по штатам
plt.figure(figsize=(12, 6))
data['State'].value_counts().plot(kind='bar')
plt.title('Количество абонентов по штатам')
plt.xlabel('Штат')
plt.ylabel('Количество абонентов')
plt.show()

# Парные графики для числовых признаков и факта оттока
num_feats = ['Total day charge', 'Total intl charge', 'Customer service calls']
sns.pairplot(data[num_feats + ['Churn']], hue='Churn')
plt.suptitle('Парные графики для числовых признаков и факта оттока', y=1.02)
plt.show()

# Scatter plot для двух признаков "Total day charge" и "Total intl charge" с раскрашиванием по факту оттока
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total day charge', y='Total intl charge', hue='Churn', data=data, palette='coolwarm')
plt.title('Scatter plot для Total day charge и Total intl charge')
plt.xlabel('Дневные начисления')
plt.ylabel('Международные начисления')
plt.show()
churn_index = data.columns.get_loc("Churn")

# Оставляем только признаки до "Churn"
data_processed = data.iloc[:churn_index]
data_processed = data_processed.drop(columns=['International plan', 'Voice mail plan', 'State'])

# Преобразуем категориальные переменные в числовой формат с помощью One-Hot Encoding
data_processed = pd.get_dummies(data_processed)

# Тепловая карта корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(data_processed.corr(), cmap=plt.cm.Blues)
plt.title('Тепловая карта корреляции ')
plt.tight_layout()
plt.show()



