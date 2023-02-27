# Databricks notebook source
import pandas as pd 

# COMMAND ----------

df = pd.DataFrame({
    'a':[1,2,3],
    'b':[4,5,6],
    'c':[7,8,9],
}, index = [1,2,3])


#df2 = pd.DataFrame(data, index, columns)
df2 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], 
      index=[1,2,3], 
      columns = ['a','b','c'])

print(df)
print("break")
print(df2)

# COMMAND ----------

print(df)
print("break")
print(df.T)

# COMMAND ----------

tipsData = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
#print(tipsData.head(5))

print(tipsData.describe())

print(tipsData.isnull().sum())

# COMMAND ----------

tipsData.groupby(['day']).sum()


# COMMAND ----------

tipsData.groupby(['day']).count()

# COMMAND ----------

totalTips = tipsData.groupby(['day']).sum()['tip']
totalBill = tipsData.groupby(['day']).sum()['total_bill']

tipDayPercentage = (100 * totalTips / totalBill)

tipDayPercentage = tipDayPercentage.to_frame('tip(%)').reset_index()

print(tipDayPercentage)

# COMMAND ----------

pd.DataFrame({'Apples': ['30'], 'Bannas': ['21']})

# COMMAND ----------

pd.DataFrame({'Apples': ['35', '41'], 
              'Bannas': ['21','34']},
            index = ['2017 Sales', '2018 Sales'])

# COMMAND ----------

import pandas as pd

pd.Series([4, 1, 2, 1], index =['Flour', 'Milk', 'Eggs', 'Spam'], name = 'Dinner')

# COMMAND ----------

import pandas as pd

#Series in Pandas
s = pd.Series(dtype='float64')
s

# COMMAND ----------

#Series using NumPy Array
import pandas as pd
import numpy as np 

d = np.array([1,2,3,4,5])
s = pd.Series(d)
s

# COMMAND ----------

#Series using list
d = [1,2,3,4,5]
s = pd.Series(d)
s

# COMMAND ----------

#Series with index

d = [1,2,3,4,5]
s = pd.Series(d, index = ["one", "two", "three", "four", "five"])
s

# COMMAND ----------

#Series using Dictionary 

d = {"one" : 1, 
    "two" : 2,
    "three" : 3, 
    "four" : 4,
    "five" : 5}

s = pd.Series(d)
s

# COMMAND ----------

#Series using Scalar value

s = pd.Series(1, index = ["a", "b", "c", "d"])
s

# COMMAND ----------

#Dataframe in Pandas

#Creating an empty DataDrame
df = pd.DataFrame()
print(df)

# COMMAND ----------

#Creating a DataFrame using list

list = ["Manufacturing Level", "Machining", "Treatments", "Assembly"]
df = pd.DataFrame(list)
print(df)

# COMMAND ----------

#Creating DataFrames using Dictonary of ndarry/Lists

batmanData = {'Movie Names' : ['Batman Begins', 'Dark Knight', 'The Dark Knight Rises'], '    Year of Release': [2005, 2008, 2012]}
df = pd.DataFrame(batmanData)
print(df)                               

# COMMAND ----------

#Creating DataFrames using Dictonary of ndarry/Lists (Work)
MachineData = {' Machine Name' : ['Handtmann 1', 'Handtmann 2', 'Handtmann 3'], '   Machine Number' : ['M/C 091', 'M/C 092', 'M/C 093'], '   Year of Manufacture' : ['2009', '2009', '2009']}
df = pd.DataFrame(MachineData)
print(df)

# COMMAND ----------

#Creating DataFrame using List of Lists [Numbers]

data = [['Alex', 601], ['Bob', 602], ['Cataline', 603]]
df = pd.DataFrame(data, columns = ['Names', '  Roll No.'])
print(df)

# COMMAND ----------

#Creating DataFrame using List of Lists [Numbers - Work]

data = [['Structures', 1], ['Treatments', 2], ['Pipe Cell', 3], ['Davy Way', 4]]
df = pd.DataFrame(data, columns = [' Site Name', '  Site Number'])
print(df)

# COMMAND ----------

#Creating a DataFrame using Zip() fucntion

Name = ['Alex', 'Bob', 'Cataline']
RollNo = [601, 602, 603]

listOfTuples = list(zip(Name, RollNo)
                    
df = pd.DataFrame(listOfTuples, columns = ['Name', 'Roll No.'])
print(df)

# COMMAND ----------

import pandas as pd
Name = ['Alex', 'Bob', 'Cataline']
RollNo = [601, 602, 603]

listOfTuples = list(zip(Name, RollNo))

df = pd.DataFrame(listOfTuples, columns = ['Name', '  Roll No.'])
print(df)

# COMMAND ----------

#How to read data in Pandas (All data)
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Yuvrajchandra/Basic-Operations-Using-Pandas/main/biostats.csv')
print(df.to_string())

# COMMAND ----------

#How to read data in Pandas (Select rows)

df = pd.read_csv('https://raw.githubusercontent.com/Yuvrajchandra/Basic-Operations-Using-Pandas/main/biostats.csv')
print(df.head())

#The df.head() signifies that the first 5 rows will be shown. 

# COMMAND ----------

#How to read data in Pandas (Select rows)

df = pd.read_csv('https://raw.githubusercontent.com/Yuvrajchandra/Basic-Operations-Using-Pandas/main/biostats.csv')
print(df.head(10))

#The df.head(10) signifies that the first 10 rows will be shown. 

# COMMAND ----------

#How to read data in Pandas (Select rows)

df = pd.read_csv('https://raw.githubusercontent.com/Yuvrajchandra/Basic-Operations-Using-Pandas/main/biostats.csv')
print(df.tail())

#The df.tail() signifies that the last 5 rows will be shown. 

# COMMAND ----------

#How to read data in Pandas (Select rows)

df = pd.read_csv('https://raw.githubusercontent.com/Yuvrajchandra/Basic-Operations-Using-Pandas/main/biostats.csv')
print(df.tail(10))

#The df.tail(10) signifies that the last 10 rows will be shown. 
