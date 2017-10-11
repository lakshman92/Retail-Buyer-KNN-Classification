# Script 4 3 for students  - KNN implementation from Textbook
# This contains commands covered by Viswanathan Lab 8 (p. 136-147)
# Table 35 on p. 147 contains the detailed runthrough

# You can run this script or type in the commands on your own

# Step 0:  Get ourselves into the Week 4 directory
# Also copy the "vacation-trip-classification.csv" file to this directory

getwd()
cat("We have changed to C:/R-WorkSpace, and here's the list of files.")
cat("Make sure your 'vacation-trip-classification.csv' file is in there")
list.files()


# Step 1:  Read the data into a frame called vac
vac <- read.csv("vacation 500 v02.csv")

cat("Step 1:  Here is the head of your vacation trip classification data")
head(vac)

# Step 2:  Understand and explore the data
boxplot(vac$Income)
hist(vac$Income)
boxplot(vac$Family_size)
hist(vac$Family_size)
boxplot(Income ~ Result, data = vac, ylab="Income")
boxplot(Family_size ~ Result, data = vac, ylab = "Family Size")

# Question:  which variables are our independent variables?
# Which variable is our target variable?


# Step 3:  standardize the variables
vac$Income_z <- scale(vac$Income)
vac$Family_size_z <- scale(vac$Family_size)
head(vac)

# Step 4:  create three partitions:  train.a, train.b, and test

# load the caret package - check in 'packages' to the bottom right that it loaded
library(caret)

set.seed(24)
samp <- createDataPartition(vac$Result, p = 0.6, list = FALSE)
train.a <- vac[samp,]
rest <- vac[-samp,]
samp <- createDataPartition(rest$Result, p = 0.5, list = FALSE)
train.b <- rest[samp,]
test <- rest[-samp,]
nrow(vac)
nrow(train.a)
nrow(train.b)
nrow(test)

# who is in which partition?
train.a
head(train.b)
head(test)

# Step 5 - be clear about our model parameters
# In this case, columns 4 and 5 in train.a and train.b contain the 
#      scaled independent variables we want (Income_z and Family_size_z)
train.a[,4:5]

# Column 3 in train.a contains the outcome variable (buyer / non-buyer)
train.a[,3]

# Step 6 - load the 'class' package
library(class)
# Click on the bottom right window, 'Packages' menu, to verify the class library loaded

# Here we start our model building: we are going to iterate through 
# k=1, k=3, k=5, etc. until we find our error matrix performance declines

# Step 7 - build model for k=1

#if it spoke English, we'd tell it to do something like this:
#Use the Income_z and Family_size_z from train.a, and the Income_z and Family_size_z from train.b, 
# and the Result (buyer or non-buyer) from train.a, and k=1 to run the KNN algorithm 
# and put all the predicted results (buyer or non-buyer) in train.b$pred.1 

# but it speaks R instead, so we need to talk to it like this
train.b$pred.1 <- knn(train.a[,4:5],train.b[,4:5], train.a[,3], 1)

# It should have added a 'pred.1' column to the far right of train.b
# the 'pred' stands for predicted, and the '1' reminds us this was for k=1

head(train.b)

# Step 8:  Compute and display the error matrix for k = 1
tab.1 <- table(train.b$Result, train.b$pred.1, dnn = c("Actual", "Predicted"))
tab.1

# Step 9:  Let's try it again for k = 3
train.b$pred.3 <- knn(train.a[,4:5],train.b[,4:5], train.a[,3], 3)
tab.3 <- table(train.b$Result, train.b$pred.3, dnn = c("Actual", "Predicted"))
tab.3

# Step 9 still:  Let's try it again for k = 5
train.b$pred.5 <- knn(train.a[,4:5],train.b[,4:5], train.a[,3], 5)
tab.5 <- table(train.b$Result, train.b$pred.3, dnn = c("Actual", "Predicted"))
tab.5

# Step 9 still:  Let's try it again for k = 7
train.b$pred.7 <- knn(train.a[,4:5],train.b[,4:5], train.a[,3], 7)
tab.7 <- table(train.b$Result, train.b$pred.3, dnn = c("Actual", "Predicted"))
tab.7

# Step 9 finally:  Let's try it again for k = 9 (doesn't seem to be getting better)
train.b$pred.9 <- knn(train.a[,4:5],train.b[,4:5], train.a[,3], 9)
tab.9 <- table(train.b$Result, train.b$pred.3, dnn = c("Actual", "Predicted"))
tab.9

# Step 9 grand finale:  let's see if k=1 or k=3 is better
prop.table(tab.1)
prop.table(tab.3)

# K=1 classifies better, so I'll go with the k=1 model

# Step 10:  
# If it spoke English, we'd tell it something like this:
# Use the Income_z and Family_size_z from train.a, and the Income_z and Family_size_z from test, 
# and the Result (buyer or non-buyer) from train.a, and a k=1 to run the KNN algorithm 
# and put all the predicted results (buyer or non-buyer) in test$pred.1 

# but it speaks R, so we need to tell it this instead:
test$pred.1 <- knn(train.a[,4:5],test[,4:5], train.a[,3], 1)

# Let's see how it did - there should be a new pred.1 column on the right side of test
head(test)

# Step 11 - generate error matrix for our test data

#tab.test.3 <- table(test$Result, test$pred.1, dnn=c("Actual", "Predicted"))
#tab.test.3
#prop.table(tab.test.3)

tab.test.1 <- table(test$Result, test$pred.1, dnn=c("Actual", "Predicted"))
tab.test.1
prop.table(tab.test.1)


# Last step:  let's report the results
# To get the lift, we need to know our naive classification
# Naive classification:  how many buyers/non-buyers did our original data set have? 

summary(vac$Result)

# OK, so there were 21 buyers out of 40.  With no model, we'd assume everybody was a buyer,
# for a naive success rate of 21/40 = 52.5%.

# How did our model with k=1 do on our test data?
prop.table(tab.test.1)

# This says we clasified 28.5% of buyers correctly + 28.5% of non-buyers = about 57% correct.
# Lift is our model / naive classification
# which is 57% / 52.5% = 1.09.
#
# Our lift is greater than 1, which means this model is useful at k=1.
# Hooray for us!

