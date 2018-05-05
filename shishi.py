import pandas as pd
D = {-1:([4,'s'],[5])}
C = {}
B = []
def a(t=[],p=[]):
    return t,p

# D[1] = a([1,2], [3,4])
print(a([1,2], [3,4]))
print(D[1][1])
for i in D[1]:
    B.append(pd.Series(i))

print(pd.concat(B))

print('=================')
L = [1,]
L.append('s')
print(L)