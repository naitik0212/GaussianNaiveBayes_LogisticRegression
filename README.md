# GaussianNaiveBayes_LogisticRegression
Implementation from Scratch using Lists, Pandas and Numpy Array
As the Gaussian Naïve Bayes and Logistic Regression are implemented, now running it across 5 runs with increasing training data. The ratio of training data passed is [0.01,0.02,0.05,0.1,0.625,1] in random fractions(i.e. different training points each time). As we can observe in this graph, as the training data increases, the testing accuracy increases up to a certain point and then remains constant.

The highest recorded accuracy for Gaussian Naïve Bayes is around 86% while for Logistic Regression the maximum accuracy is around 96.5%. We can infer that, for the given Bank_Data, Logistic regression correctly predicts the test class with higher accuracy compared to Gaussian Naïve Bayes.

Random Sample Generation:
The function generatedSample took the parameters of mean and standard deviation and generated 400 new samples using the function "np.random.normal". The mean and variance (using numpy mean and numpy variance) along the columns were calculated for each features and appended in a list and finally displayed.

Cross Validation was performed on the given Bank_Data by iterating over a loop to find the Mean and Variance of each model. I have divided the different traindata as 67% and test data as 33% for each runs by using the functions splitDataset and MergeListExcept(merges two parts except the ith part for test data.)
