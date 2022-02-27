import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-whitegrid')

names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship',
         'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week', 'Country', 'income']

# 获取原始数据集
adult_data = pd.read_csv('../data/adult.csv', names=names)
print('原始数据集', adult_data.shape)
print(adult_data.head())


def laplace_mech(v, s, epsilon):
  return v + np.random.laplace(0, s / epsilon)


def pct_error(origin, priv):
  return np.abs(origin - priv) / origin * 100.0


options = adult_data['Marital_Status'].unique()
print('options result: ', options)


def score(data, option):
  # print(data.value_counts())
  return data.value_counts()[option] / data.shape[0]
  # return data.value_counts()[option] / 1000


print('打印：Never-married score: ', score(adult_data['Marital_Status'], ' Never-married'))


def exponential(dataset, R, u, sensitivity, epsilon):
  # Calculate the score for each element of R
  scores = [u(dataset, r) for r in R]
  print(scores, len(scores), np.sum(scores))
  
  # Calculate the probability for each element, based on its score
  probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
  print(probabilities, len(probabilities))
  
  # Normalize the probabilties so they sum to 1
  probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
  print(probabilities, len(probabilities), np.sum(probabilities))
  
  # Choose an element from R based on the probabilities
  return np.random.choice(R, size=1, replace=False, p=probabilities)


r = exponential(adult_data['Marital_Status'], options, score, 1, 1)
print(pd.Series(r).value_counts())


r = [exponential(adult_data['Marital_Status'], options, score, 1, 1) for i in range(200)]
print(pd.Series(r).value_counts())