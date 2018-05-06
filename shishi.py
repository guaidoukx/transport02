import pandas as pd

class Stu(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def add(self, t):
        L = [t]
        self.L = L
        
Astu = Stu('A', 2)
print(Astu.L)
Astu.add(3)
print(Astu.L)