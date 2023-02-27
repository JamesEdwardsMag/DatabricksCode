# Databricks notebook source
import numpy as np

# COMMAND ----------

my1dArray = np.array([1,2,3])
my2dArray = np.array([[1,2,3], [4,5,6]], dtype = float)
my2dArray = np.array(((1,2,3), (4,5,6)), dtype = np.float32)

charArray1 = np.array([(1,'a', 3.), (4,5, 'zzz')], dtype = "U21")

print(charArray1)

# COMMAND ----------

rows = 4
cols = 3

array1s = np.ones((rows, cols))

array5s = 5 * np.ones((rows, cols))

array7s = np.full((rows, cols), 7)

print(array5s)
print(array7s)

# COMMAND ----------

samples = 3
step = 3
start = 10
stop = 25

np.arange(start, stop, step)

np.linspace(start, stop, step)

np.eye(3)

np.random.random((2,3))

np.random.randint(0,10, (3,3))

# COMMAND ----------

a = np.array([[1,2,3], 
              [4,5,6], 
              [7,8,9]])

a.ndim #dimensions
a.shape
a.size
a.dtype #data type

#a[0,1]
a[a < 9]

np.diag(a) #diagonal

# COMMAND ----------

a = np.array([[1,2], [4,5]])
b = np.array([[9,8], [6,5]])

summ1 = a.sum()
summ2 = np.sum(a)

a + b
np.add(a,b)

a - b
a * b
a / b

np.exp(a)

np.sqrt(a)

np.sin(a)
np.cos(a)
np.log(a)

display(a.dot(b))

#print(summ1, summ2)

# COMMAND ----------

# (0,3)
# |\
# | \
# |  \
# |___\ (1.5, 0)
# (0,0)

A = np.array([0,0])
B = np.array([0,3])
C = np.array([1.5,0])


# Comput AB

# AB^2 = (ax-bx)^2 + (ay-by)^2
# A^2 + B^2 = C^2

AB_diffSquared = np.power(A-B, 2)
AB_squared = np.sum(AB_diffSquared)
AB = np.sqrt(AB_squared)

# Comput AC

AC = np.sqrt(np.sum(np.power(A-C,2)))

# Comput BC

BC = np.sqrt(AB**2 + AC**2)


#Easier way of doing the maths...

from numpy import linalg as la

AB2 = la.norm(A-B)

AC2 = la.norm(A-C)

BC2 = la.norm(B-C)

#Check
print (AB2, AB)
print (AC2, AC)
print (BC2, BC)
