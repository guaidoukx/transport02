import pandas as pd
D = {-1:([4,'s'],[5])}
print(D[-1][0])
C = ['a']
B = ['c','d']
print(pd.concat([pd.Series(B)]))
m = C+B
print(m)
C1 = pd.concat([pd.Series(C+B)])
print(C1)
# def a(t=[],p=[]):
#     return t,p
#
# # D[1] = a([1,2], [3,4])
# print(a([1,2], [3,4]))
# print(D[-1][1])
# for i in D[-1]:
#     B.append(pd.Series(i))
#
# print(pd.concat(B))
# B.append(['mdn'])
# print(B)
# print('=================')
# L = [1,]
# L.append('s')
# print(L)