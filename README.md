# GaussianNaiveBayes_LogisticRegression
Implementation from Scratch using Lists, Pandas and Numpy Array

Gausian Naive Bayes
Naive Bayes can be extended to real-valued attributes, most commonly by assuming a Gaussian distribution.

This extension of naive Bayes is called Gaussian Naive Bayes. Other functions can be used to estimate the distribution of the data, but the Gaussian (or Normal distribution) is the easiest to work with because you only need to estimate the mean and the standard deviation from your training data.

As the Gaussian Naïve Bayes and Logistic Regression are implemented, now running it across 5 runs with increasing training data. The ratio of training data passed is [0.01,0.02,0.05,0.1,0.625,1] in random fractions(i.e. different training points each time). As we can observe in this graph, as the training data increases, the testing accuracy increases up to a certain point and then remains constant.

The highest recorded accuracy for Gaussian Naïve Bayes is around 86% while for Logistic Regression the maximum accuracy is around 96.5%. We can infer that, for the given Bank_Data, Logistic regression correctly predicts the test class with higher accuracy compared to Gaussian Naïve Bayes.

Random Sample Generation:
The function generatedSample took the parameters of mean and standard deviation and generated 400 new samples using the function "np.random.normal". The mean and variance (using numpy mean and numpy variance) along the columns were calculated for each features and appended in a list and finally displayed.

Cross Validation was performed on the given Bank_Data by iterating over a loop to find the Mean and Variance of each model. I have divided the different traindata as 67% and test data as 33% for each runs by using the functions splitDataset and MergeListExcept(merges two parts except the ith part for test data.)

Logistic regression is another technique borrowed by machine learning from the field of statistics.

It is the go-to method for binary classification problems (problems with two class values). In this post you will discover the logistic regression algorithm for machine learning.


References: https://machinelearningmastery.com/naive-bayes-for-machine-learning/
            https://machinelearningmastery.com/logistic-regression-for-machine-learning/
            
