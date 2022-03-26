# 常用 numpy api

### np.random.choice(a, size=None, replace=True, p=None)
```
从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
replace:True表示可以取相同数字，False表示不可以取相同数字
数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
```

### np.dot(x,y) 
```
np.dot(x,y) == x.dot(y): x是m × n 矩阵 ，y是n×m矩阵，则x.dot(y) 得到m×m矩阵。
```