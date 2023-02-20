# Databricks notebook source
print("hello world")

# COMMAND ----------

#Strings
first_name = "James"
second_name = "Edwards"

#Booleans
bool1 = True
bool2 = False

#Integers
int1 = 10
int2 = 100
int3 = -100

#floats
float1 = 1.5
float2 = -1.5

#list
list1 = [50, 100, 150, 200, 250]
list_product = ["clock", "car", "bus"]

print(list_product)

print(first_name + " " +second_name)

# COMMAND ----------

number1 = input("please enter number")
number2 = input("enter a second")

print(int(number1)+int(number2))


# COMMAND ----------

varName = "variable"
var2Name = 12

var2Name -=2

print(var2Name)

# COMMAND ----------

num1 = 18
num2 = 14

#remove "#" to run programme.

#print(num1 == num2) #the same as
#print(num1 < num2) #num2 is more than num1
#print(num1 > num2) #num1 is more than num2
#print(num1 != num2) #not equal to
#print(num1 <= num2) #num2 is more than num1
#print(num1 >= num2) #num1 is more than num2

#if num1 > num2:
    #print("first number and all that")
#elif num1 == num2:
    #print("The numbers are equal")
#else:
    #print("The second number is greater")

# COMMAND ----------

#Task 1 
# Ask for two numbers to be input, then display the larger of the two. 

num1 = 50
num2 = 10

if num1 > num2:
    print(num1)

if num1 < num2:
    print(num2)

# COMMAND ----------

#Task 2
# Write some code that will output to the screen if the year you were born was a leap year


#year = 1998

#if(year % 400 == 0) and (year % 4 == 0):
    #print("Is a leap year")
    
#elif(year % 4 == 0) and (year % 100 != 0):
    #print("Is a leap year")
    
#else:
    #print("Is a not leap year")
    
    
inputYear = input("when you were born")    

if(int(inputYear)% 4 == 0):
    print("Born on a leap year")
    
else:
    print("Not born on a leap year")

# COMMAND ----------

#           0,1,2,3,4,5,6,7,8,9,10
products = [1,2,3,4,"hi",6,7,8,8,79,18]


for counter in range(len(products)):
    print(counter, products[counter])


# COMMAND ----------

count = 1 
while count < 10:
    print(count)
    count +=1

# COMMAND ----------

password = "opennow1!"

entry = False

while entry != password:
    entry = input("please enter your password")
    
print("welcome")

# COMMAND ----------

#Task 3 
#Ask the user to enter a series of numbers

#Create a total of adding each number to the last

#Stop adding numbers when the user types zero

#Print out the total at the end 


total = 0 

while True:
    guess = input("next number?...")
    if(int(guess) == 0):
        break
    else:
        total+=int(guess)
        print("current total: ", total)
        
print (total)

# COMMAND ----------


