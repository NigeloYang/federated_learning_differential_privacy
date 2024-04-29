import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-whitegrid')

adult = pd.read_csv('./data/adult_with_pii.csv')
print('原始数据集', adult.shape)
print(adult.head(2))


def laplace_mech(v, sensitivity, epsilon):
  return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)


def pct_error(orig, priv):
  return np.abs(orig - priv) / orig * 100.0


# AboveThreshold Algorithm
def above_threshold(df, queries, T, epsilon):
  T_hat = T + np.random.laplace(loc=0, scale=2 / epsilon)
  
  for idx, q in enumerate(queries):
    nu_i = np.random.laplace(loc=0, scale=4 / epsilon)
    if q(df) + nu_i >= T_hat:
      return idx
  return -1  # the index of the last element


def age_sum_query(df, b):
  return df['Age'].clip(lower=0, upper=b).sum()


print('search age sum:', age_sum_query(adult, 30))


def naive_select_b(df, query, epsilon):
  bs = range(1, 120, 10)
  best = 0
  threshold = 10
  epsilon_i = epsilon / len(bs)
  
  for b in bs:
    r = laplace_mech(query(df, b), b, epsilon_i)
    print('r result:', r)
    
    # if the new answer is pretty close to the old answer, stop
    if r - best <= threshold:
      print('r = {}  best = {}'.format(r, best))
      return b
    # otherwise update the "best" answer to be the current one
    else:
      best = r
  
  return bs[-1]


print(naive_select_b(adult, age_sum_query, 1))


# 确定使用稀疏向量技术的b最佳值的索引
def create_query(b):
  return lambda df: age_sum_query(df, b) - age_sum_query(df, b + 1)


bs = range(1, 150, 5)
queries = [create_query(b) for b in bs]
epsilon = .1

print('above_threshold result: ', bs[above_threshold(adult, queries, 0, epsilon)])

plt.xlabel('Chosen Value of "b"')
plt.ylabel('Number of Occurrences')
plt.hist([bs[above_threshold(adult, queries, 0, epsilon)] for i in range(20)]);
plt.show()


# 只需要找到第一个大于阈值的索引就可以返回结果，也就高于阈值算法: AboveThreshold
def auto_avg(df, epsilon):
  def create_query(b):
    return lambda df: df.clip(lower=0, upper=b).sum() - df.clip(lower=0, upper=b + 1).sum()
  
  # Construct the stream of queries
  bs = range(1, 150000, 5)
  queries = [create_query(b) for b in bs]
  
  # Run AboveThreshold, using 1/3 of the differential_privacy budget, to find a good clipping parameter
  epsilon_svt = epsilon / 3
  final_b = bs[above_threshold(df, queries, 0, epsilon_svt)]
  
  # Compute the noisy sum and noisy count, using 1/3 of the differential_privacy budget for each
  epsilon_sum = epsilon / 3
  epsilon_count = epsilon / 3
  
  noisy_sum = laplace_mech(df.clip(lower=0, upper=final_b).sum(), final_b, epsilon_sum)
  noisy_count = laplace_mech(len(df), 1, epsilon_count)
  
  return noisy_sum / noisy_count


print('Age result:', auto_avg(adult['Age'], 1))
print('Capital_Gain result:', auto_avg(adult['Capital Gain'], 1))


# 找到大于所有索引的值 也就是稀疏算法：Sparse Algorithm
def sparse(df, queries, c, T, epsilon):
  idxs = []
  pos = 0
  epsilon_i = epsilon / c
  
  # stop if we reach the end of the stream of queries, or if we find c queries above the threshold
  while pos < len(queries) and len(idxs) < c:
    # run AboveThreshold to find the next query above the threshold
    next_idx = above_threshold(df, queries[pos:], T, epsilon_i)
    
    # if AboveThreshold reaches the end, return
    if next_idx == -1:
      return idxs
    
    # otherwise, update pos to point to the rest of the queries
    pos = next_idx + pos
    # update the indices to return to include the index found by AboveThreshold
    idxs.append(pos)
    # and move to the next query in the stream
    pos = pos + 1
  
  return idxs


epsilon = 1
print(sparse(adult['Age'], queries, 3, 0, epsilon))