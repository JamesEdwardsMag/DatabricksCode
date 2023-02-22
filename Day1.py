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

#Day 1 - Task 1 
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

#Day 2 - Task 1 
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

import random

num = 3.1473

#print(eval("num+1"))

#print(random.randint(1,100))

# COMMAND ----------


#parameter/ argument
def printmessage(message):
    print(message)

        
printmessage("hello")
printmessage("goodbye")



# COMMAND ----------

def square(number):
    squarenumber = number ** 2 #squared power
    return squarenumber

numtosquare = 4

answer = square(numtosquare)

print("the square of "+str(numtosquare) + " is", answer)


# COMMAND ----------

#Task 3
#lbs to Kg

lbs = float(input("Enter Lbs"))
Kg = 0.45359237

Answer = lbs*Kg
print("Kg", Answer)


# COMMAND ----------

#Task 3
#Kg to lbs

Kg = float(input("Enter Kg"))
lbs = 2.204

Answer = Kg*lbs
print("lbs", Answer)


# COMMAND ----------

#Task 3
#Celcius to Fahrenheit

Celcius = float(input("Enter Celcius"))
Fahrenheit = ((Celcius *9/5) +32)

Answer = Fahrenheit
print("Fahrenheit", Answer)

# COMMAND ----------

#Task 3
#Fahrenheit to Celcius

Fahrenheit = float(input("Enter Fahrenheit"))
Celcius = ((Fahrenheit -32) *5/9)

Answer = Celcius
print("Celcius", Answer)

# COMMAND ----------

my_list = ["i", "d", "l", "e"]

#del my_list[2:4]
#my_list[1] = "n"
#new_list = ["s"] + my_list
#new_list.append("g")
#print(new_list)

final_list = []
for i in my_list:
    print(i, my_list.index(i))
    final_list.append(my_list.index(i))
    
print(final_list)

print(sum(final_list))
print(len(final_list))


# COMMAND ----------

# Create a list that contains ten integers your user has typed in at the keyboard.
user_input = []

#for i in range(10):
#    print(i)
#    input_number = input("input a number")

user_input = [1,2,3,4,5,6,7,8,9,0]

# Calculate & display the sum and average of the numbers stored in this new list.

input_sum = sum(user_input)
input_average = input_sum/ len(user_input)

# Display each number in turn, along with a message stating whether it is below, above or equal to the average

for number in user_input:
    print(number)
    if number == input_average:
        print("Number is equal to average")
    elif number < input_average:
        print("Number is less than average")
    else:
        print("Number is greater than average")

# e.g. 10 is above average

# COMMAND ----------

Length = float(input("Enter Length"))
Width = float(input("Enter Width"))

Answer = (Length + Width)*2
print("Perimeter =", Answer,"m")

# COMMAND ----------

#Day 3 - Task 1

x = "yes"
while x == "yes":
    Roll = random.randint(1,6)
    print(Roll)
    
    if Roll == 1:
        print(" _____ ")
        print("|     |")
        print("|  0  |")
        print("|     |")
        print(" ----- ")
        
    if Roll == 2:
        print(" _____ ")
        print("|0    |")
        print("|     |")
        print("|    0|")
        print(" ----- ")
        
    if Roll == 3:
        print(" _____ ")
        print("|0    |")
        print("|  0  |")
        print("|    0|")
        print(" ----- ")
   
    if Roll == 4:
        print(" _____ ")
        print("|0   0|")
        print("|     |")
        print("|0   0|")
        print(" ----- ")
        
    if Roll == 5:
        print(" _____ ")
        print("|0   0|")
        print("|  0  |")
        print("|0   0|")
        print(" ----- ")
        
    if Roll == 6:
        print(" _____ ")
        print("|0 0 0|")
        print("|     |")
        print("|0 0 0|")
        print(" ----- ")
   
    x=input("Roll again? Type yes. Or type no to exit:")
    print("\n")
    

# COMMAND ----------

#Day 3 - Task 1

x = "yes"
while x == "yes":
    Roll = random.randint(1,6)
    print(Roll)
    
    x=input("Roll again? Type yes or type no to exit:")
    print("\n")

    

# COMMAND ----------

#Day 3 - Task 2

import random
num = random.randint(0,100)

while guess != num:
    guess =int(input("Guess the number"))
    
    if guess == num:
           print("Good guess!")
                            
    elif guess < num:
           print("Higher...") 
                    
    else:
           print("Lower...")
  
