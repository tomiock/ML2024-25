r"""°°°
<a href="https://colab.research.google.com/github/dkaratzas/ML2024-25/blob/main/Session%202%20-%20Multiple%20and%20Polynomial%20regression/P2_Multiple_and_Polynomial_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
°°°"""
# |%%--%%| <8tqhMN6dFW|1FB8bSeHzU>
r"""°°°
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/dkaratzas/ML2024-25/blob/main/Session%202%20-%20Multiple%20and%20Polynomial%20regression/P2_Multiple_and_Polynomial_Regression.ipynb)
°°°"""
# |%%--%%| <1FB8bSeHzU|a0bOBMsnLR>
r"""°°°
# Problems 2 - Polynomial Regression
°°°"""
# |%%--%%| <a0bOBMsnLR|douVnkkQUw>
r"""°°°
## 2.1 Normalization (Standardisation)

Imagine you are given 20 samples with the following values for a feature x:
°°°"""
# |%%--%%| <douVnkkQUw|lS2869sE9s>

import numpy as np

x1 = np.array([56.73040025,  42.07755103,  83.46673661, 167.79511467,
       128.41593193, 1620.39079195,  67.57569387, 124.50199413,
        91.58831309, 241.75090834,  93.69791353,  80.43787513,
        20.38769615, 171.08315486,  59.2372586 ,  15.73475339,
       122.8934116 ,  27.26541632, 217.80864704, 150.23539618])

x2 = np.array([ 0.11179419,  0.22728357, -0.08936106,  0.59369292,  0.36202046,
        0.66546626, -0.1019957 ,  0.63595947, -0.10978375,  0.68933564,
        0.05552168,  0.5122346 ,  0.04727783, -8.53652367,  0.60681752,
        0.66415377,  0.68607407,  0.15515183, -0.16490555,  0.77765625])

# |%%--%%| <lS2869sE9s|9C74XQ6aVW>
r"""°°°
Note that the ranges of these two features look very different
°°°"""
# |%%--%%| <9C74XQ6aVW|ZW6PjYtYvx>

print('Original range for feature 1: [{0}, {1}]'.format(x1.min(), x1.max()))
print('Original range for feature 2: [{0}, {1}]'.format(x2.min(), x2.max()))

# |%%--%%| <ZW6PjYtYvx|KAWEXb8s6m>
r"""°°°
<br>

First of all lets plot them along one dimension, to see what their distribution looks like. To do this, we will do a scatter plot, but will set all Y coordinates to zero.
°°°"""
# |%%--%%| <KAWEXb8s6m|B3wKuCtVvL>

import matplotlib.pyplot as plt
#                            use the same X axis
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

ax1.scatter(x1, np.zeros_like(x1), color = 'g', marker ='.', alpha=0.5)
ax2.scatter(x2, np.zeros_like(x2), color = 'b', marker ='.', alpha=0.5)

# |%%--%%| <B3wKuCtVvL|KM9kub4A8m>
r"""°°°
The two features are in very different ranges, while it seems that there is one value which is quite off on the right on the top plot. This is probably an *outlier*, an error in measurement for example.

<font color=blue>Find the range of this data and normalise the data to $[0, 1]$ by subtracting the minimum value and dividing by the range you find.</font>
°°°"""
# |%%--%%| <KM9kub4A8m|aCUMkET1sk>

# Your Code Here #



# |%%--%%| <aCUMkET1sk|vyoFCWZId6>
r"""°°°
<font color=blue>What do you observe? What is the effect of the outlier in the data?</blue>
°°°"""
# |%%--%%| <vyoFCWZId6|G2NAcXFh5X>
r"""°°°
---

*Answer*

Basically, it makes the range of this features very large, which can lead to some problems down the line.
---

°°°"""
# |%%--%%| <G2NAcXFh5X|lhMDu8uPC7>
r"""°°°
<font color=blue>Now normalize by centering the data and dividing with the standard deviation. Plot the scaled data again, what is the range? What is the effect of scaling your data like this?</font>
°°°"""
# |%%--%%| <lhMDu8uPC7|i3dAnmvXA4>

# Your Code Here #



# |%%--%%| <i3dAnmvXA4|nciAhbbwyx>
r"""°°°
---

*Answer*

*YOUR ANSWER HERE*

---

°°°"""
# |%%--%%| <nciAhbbwyx|jPmp12vGJ8>
r"""°°°
We now want to remove any outliers. We will consider anything above 1000 in feature 1 or anything below -5 in feature 2 as an outlier.

<font color=blue>Can you think of a way to remove the outliers? An easy way to identify these values is by using masks to index.</font>
°°°"""
# |%%--%%| <jPmp12vGJ8|ClGTbcUBYE>

# Your Code Here #

mask = (x1 <= 100 & x2 > -5)

x1_new = x1[mask]
x2_new = x2[mask]

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)

ax1.scatter(x1_new, np.zeros_like(x1_new), color = 'g', marker ='.', alpha=0.5)
ax2.scatter(x2_new, np.zeros_like(x2_new), color = 'b', marker ='.', alpha=0.5)

# |%%--%%| <ClGTbcUBYE|g91JKMkMka>
r"""°°°
## 2.2 Multiple Linear Regression with Normalisation

Let's revisit the problem of house prices that we saw last week. The task to be done is to predict the price of a house given some 'features' of the house. Our input data comprises two features per sample (size of the house, and number of rooms), while the output data is the price of each sample.

Here is the data:

| Size (square feet) | Rooms | Price (USD) |Size (square feet) | Rooms | Price (USD) |Size (square feet) | Rooms | Price (USD) |Size (square feet) | Rooms | Price (USD) |
|:------ |:----|:----------|:------ |:----|:----------|:------ |:----|:----------|:------ |:----|:----------|
| 2104.0 | 3.0 |  **399900.0** | 1890.0 | 3.0 |  **329999.0** | 3890.0 | 3.0 |  **573900.0** | 1239.0 | 3.0 |  **229900.0** |
| 1600.0 | 3.0 |  **329900.0** | 4478.0 | 5.0 |  **699900.0** | 1100.0 | 3.0 |  **249900.0** | 2132.0 | 4.0 |  **345000.0** |
| 2400.0 | 3.0 |  **369000.0** | 1268.0 | 3.0 |  **259900.0** | 1458.0 | 3.0 |  **464500.0** | 4215.0 | 4.0 |  **549000.0** | 
| 1416.0 | 2.0 |  **232000.0** | 2300.0 | 4.0 |  **449900.0** | 2526.0 | 3.0 |  **469000.0** | 2162.0 | 4.0 |  **287000.0** |
| 3000.0 | 4.0 |  **539900.0** | 1320.0 | 2.0 |  **299900.0** | 2200.0 | 3.0 |  **475000.0** | 1664.0 | 2.0 |  **368500.0** | 
| 1985.0 | 4.0 |  **299900.0** | 1236.0 | 3.0 |  **199900.0** | 2637.0 | 3.0 |  **299900.0** | 2238.0 | 3.0 |  **329900.0** | 
| 1534.0 | 3.0 |  **314900.0** | 2609.0 | 4.0 |  **499998.0** | 1839.0 | 2.0 |  **349900.0** | 2567.0 | 4.0 |  **314000.0** | 
| 1427.0 | 3.0 |  **198999.0** | 3031.0 | 4.0 |  **599000.0** | 1000.0 | 1.0 |  **169900.0** | 1200.0 | 3.0 |  **299000.0** | 
| 1380.0 | 3.0 |  **212000.0** | 1767.0 | 3.0 |  **252900.0** | 2040.0 | 4.0 |  **314900.0** | 852.0  | 2.0 |  **179900.0** | 
| 1494.0 | 3.0 |  **242500.0** | 1888.0 | 2.0 |  **255000.0** | 3137.0 | 3.0 |  **579900.0** | 1852.0 | 4.0 |  **299900.0** | 
| 1940.0 | 4.0 |  **239999.0** | 1604.0 | 3.0 |  **242900.0** | 1811.0 | 4.0 |  **285900.0** | 1203.0 | 3.0 |  **239500.0** | 
| 2000.0 | 3.0 |  **347000.0** | 1962.0 | 4.0 |  **259900.0** | 1437.0 | 3.0 |  **249900.0** | 

Let's first get them into Python. For simplification we are going to use only one feature for the time being - the size of the house.
°°°"""
# |%%--%%| <g91JKMkMka|qBXbN6iAzO>

size = np.array([2104., 1600., 2400., 1416., 3000., 1985., 1534., 1427., 1380., 
       1494., 1940., 2000., 1890., 4478., 1268., 2300., 1320., 1236.,
       2609., 3031., 1767., 1888., 1604., 1962., 3890., 1100., 1458.,
       2526., 2200., 2637., 1839., 1000., 2040., 3137., 1811., 1437.,
       1239., 2132., 4215., 2162., 1664., 2238., 2567., 1200.,  852.,
       1852., 1203.])

rooms = np.array([3., 3., 3., 2., 4., 4., 3., 3., 3., 3., 4., 3., 3., 5., 3., 4., 2.,
       3., 4., 4., 3., 2., 3., 4., 3., 3., 3., 3., 3., 3., 2., 1., 4., 3.,
       4., 3., 3., 4., 4., 4., 2., 3., 4., 3., 2., 4., 3.])

price = np.array([399900., 329900., 369000., 232000., 539900., 299900., 314900.,
       198999., 212000., 242500., 239999., 347000., 329999., 699900.,
       259900., 449900., 299900., 199900., 499998., 599000., 252900.,
       255000., 242900., 259900., 573900., 249900., 464500., 469000.,
       475000., 299900., 349900., 169900., 314900., 579900., 285900.,
       249900., 229900., 345000., 549000., 287000., 368500., 329900.,
       314000., 299000., 179900., 299900., 239500.])

# |%%--%%| <qBXbN6iAzO|zfLYTGUrzh>
r"""°°°
The gradient descent function we used last time is the following. We have done a small modification to store and return the cost in every iteration, so that we can then plot the evolution of the cost during the optimisation (training) process.
°°°"""
# |%%--%%| <zfLYTGUrzh|HkGxwNglLL>

def GradientDescent(x, y, max_iterations=100, alpha=1):
    m = len(x) # number of samples
    J = np.zeros(max_iterations)
   
    #initialize the parameters to zero (or any other random value)
    w0 = 100000
    w1 = 100
    
    for it in range(max_iterations): #lets take a max of max_iteration steps updating the parameters
        s0 = 0 # We will use this to calculate the sum inside the cost function
        s1 = 0
        for i in range(m): #Go over the points and calculate the sum
            h = w0 + w1 * x[i]
            s0 = s0 + (h - y[i])
            s1 = s1 + (h - y[i])* x[i]
            J[it] = J[it]+(h - y[i])**2
        Grad0 = s0 / m 
        Grad1 = s1 / m 
        J[it] /= 2*m
        
        w0 = w0 - alpha * Grad0
        w1 = w1 - alpha * Grad1
        
    return [w0, w1], J
    
w, J = GradientDescent(size, price, alpha=0.00000001)

print(f'{w = }')
plt.plot(np.arange(len(J)), J)

# |%%--%%| <HkGxwNglLL|wV9NZCjiGt>
r"""°°°
<font color=blue>Use the above function with different initial values for the ws. What do you observe?</font>
°°°"""
# |%%--%%| <wV9NZCjiGt|ro6EtHUsSu>
r"""°°°
---

*Answer*

*YOUR ANSWER HERE*

---

°°°"""
# |%%--%%| <ro6EtHUsSu|5ezXDcPjHZ>
r"""°°°
<font color=blue>Now normalise your data and do the gradient descent again. What do you observe in terms of the alpha required? What do you observe in terms of the final values obtained.</font>
°°°"""
# |%%--%%| <5ezXDcPjHZ|HIr4xbzBbt>

# Your Code Here #



# |%%--%%| <HIr4xbzBbt|G10hgkAioH>
r"""°°°
---

*Answer*

Trying again the `sklearn` implementation we see how the values are similar, this is because `sklearn` also does this normalization in its processing.

---
°°°"""
# |%%--%%| <G10hgkAioH|bxZdNTDgVz>
r"""°°°
<br>

<font color=blue>Use the model you have calculated to predict the price of a house with `size = 3200` square feet.</font>
°°°"""
# |%%--%%| <bxZdNTDgVz|Y8J2D5ZvAX>

# Your Code Here #



# |%%--%%| <Y8J2D5ZvAX|x93xU75CIr>
r"""°°°
## 2.3. Vectorising our code
°°°"""
# |%%--%%| <x93xU75CIr|uqi31J12Wp>
r"""°°°
Now let's rewrite our Gradient Descent function in a slightly more efficient form, using numpy, so that it can take as input a variable number of features. X in this case will be the design matrix of size $(samples \times (features + 1) )$ and Y will be a vector of the true outputs of size $(samples \times 1)$
°°°"""
# |%%--%%| <uqi31J12Wp|VaKY7BpX5q>

def GradientDescent_np(X, y, max_iterations=100, alpha=1):
    m, n = X.shape # number of samples, number of features
    J = np.zeros(max_iterations)

    # y must be a column vector of shape m x 1
    y = y.reshape(m, 1)
    
    # Create as many parameters as features and initialize them to zero
    w = np.zeros(shape=(n, 1))
    
    # Repeat for max_iterations (it would be nice to also check convergence...)
    for iteration in range(max_iterations):
        grad = np.dot(X.T , (np.dot(X,w) - y)) / m;
        w = w - alpha*grad
        J[iteration] = ((np.dot(X,w) - y)**2).sum()
    return [w, J]

# |%%--%%| <VaKY7BpX5q|iyPaEz2ssj>
r"""°°°
Let's use the new version of Gradient Descent to calculate the parameter values 
°°°"""
# |%%--%%| <iyPaEz2ssj|mXkuqfyfAG>

#prepare the design matrix, starting with a column of ones (this is the x_0 for the bias term), 
#and concatenating the size feature

size = size.reshape(-1, 1) # Make it into a column vector of shape (m, 1)
ones = np.ones_like(size) # a column vector of ones, of the same shape as "size"

X = np.hstack( (ones, size ) )

#Let's print out a few samples to verify we have done this right
print(X[0:4, :])

# Scale features and set them to zero mean (standarize)
mu = X.mean(axis = 0) #calculate the mean for every column
sigma = X.std(axis = 0, ddof=1) #calculate the standard deviation for every column

#normalize only X1. Do not normalise X0 (the artificial first feature that we set to ones)
X[:,1] = (X[:,1] - mu[1]) / sigma[1];

#Let's print out a few samples to verify we have done this right
print(X[0:4, :])

# |%%--%%| <mXkuqfyfAG|5DV5g2JPV3>

[w, J] = GradientDescent_np(X, price, alpha=0.1)
print('w result: ', w)

plt.plot(np.arange(len(J)), J, 'bo')

# |%%--%%| <5DV5g2JPV3|TqTBuNdoxH>
r"""°°°
<font color=blue>Do the same, but this time using the rooms feature as well</font>

> Hint: Notice that the vectorised code applies on any size input - it does not matter how many samples or features you have (rows or columns in your design matrix), as it all reduces to a single matrix operation
°°°"""
# |%%--%%| <TqTBuNdoxH|ZA9vIPT3Oq>

# Your Code Here #



# |%%--%%| <ZA9vIPT3Oq|8zXbTUAFAU>
r"""°°°
## 2.4 Polynomial Regression

Suppose you are given the following samples for a problem
°°°"""
# |%%--%%| <8zXbTUAFAU|wejSeQznSN>

feat = np.array([3.70559436, 2.11187022, 2.13226786, 3.17189934, 2.61453101,
       2.07442989, 0.0071344 , 0.46131173, 3.54697197, 2.62172798,
       3.48080232, 4.77734162, 3.41456927, 0.26564345, 1.54426342,
       2.96297344, 1.17560204, 4.824855  , 4.72524112, 4.2420044 ,
       2.36161998, 4.20738357, 0.65555321, 1.54366829, 2.31498197,
       3.709236  , 2.42912614, 0.68438059, 1.71768265, 1.62213085,
       1.50209452, 0.827507  , 2.07450886, 2.24060329, 3.87450188,
       3.9819535 , 2.61195064, 2.30315148, 3.89106801, 4.43644476,
       3.37459385, 4.00239524, 4.69555677, 0.20327905, 4.37835862,
       1.38281536, 2.3788225 , 3.98380478, 3.58621116, 0.73573786])

out = np.array([30.96263192, 18.1353074 , 19.01322047, 27.51746967, 21.5564388 ,
       20.36902552,  8.62646886, 11.9312829 , 30.24023043, 21.01022706,
       26.68362214, 42.44017786, 27.8180402 , 11.75634874, 15.85208479,
       25.40096482, 12.55474498, 42.58028011, 41.31397814, 35.26195553,
       20.57609693, 36.59827885, 12.12148611, 15.50647546, 19.47160702,
       30.39893467, 20.22817055, 11.57297398, 15.01927645, 16.63212174,
       15.74075133, 12.82740801, 17.47207096, 19.72061313, 34.11748197,
       34.68450945, 21.4326298 , 19.34466379, 32.74301104, 40.46357452,
       28.37694081, 32.96050834, 41.22708261, 11.37357581, 36.19880559,
       16.2084813 , 20.18300514, 34.13550541, 31.02619715, 11.07680732])

# |%%--%%| <wejSeQznSN|Pi2AqkOcfV>
r"""°°°
<font color=blue>Make a plot of the above samples</font>
°°°"""
# |%%--%%| <Pi2AqkOcfV|Ui589jlwy7>

# Your Code Here #



# |%%--%%| <Ui589jlwy7|uu3TfjiDms>
r"""°°°
<font color=blue>Use our gradient descent function to fit a linear model to this data. Plot the resulting model along with the data. What is the final cost value?</font>
°°°"""
# |%%--%%| <uu3TfjiDms|tvVHxIG2Kb>

# Your Code Here #



# |%%--%%| <tvVHxIG2Kb|h3tMGAaN6i>
r"""°°°
<font color=blue>Now use our gradient descent function to fit a 2-degree polynomial model to the same data. Plot the resulting model along with the data. What is the final cost value?</font>
°°°"""
# |%%--%%| <h3tMGAaN6i|0JBaFqUp7h>

# Your Code Here #



# |%%--%%| <0JBaFqUp7h|i3dzReWP3E>
r"""°°°
## 2.5 Higher-degree Polynomial Regression

Suppose you are given the following samples for a problem:
°°°"""
# |%%--%%| <i3dzReWP3E|ldwWMFrxzn>

import numpy as np

# This is a single feature, with 50 data points
feat = np.array([ 3.69, -0.01,  3.92,  3.76,  4.77,  2.65, -3.2 , -3.68,  4.84,
        2.35, -3.46,  3.36,  0.05, -2.43,  4.5 ,  2.99, -4.58, -2.38,
        4.75, -0.99,  2.43,  1.03, -0.91,  0.49, -1.07, -4.56,  2.7 ,
       -1.89, -4.46, -3.22,  4.11, -1.4 , -2.15, -3.08,  1.22,  4.15,
       -3.32, -0.96,  2.28, -0.22,  1.63, -3.16,  0.57,  1.12,  0.46,
       -4.42, -1.98, -4.59, -3.27,  3.77])

# And this is the output for each of the 50 data points
out = np.array([ -4.01526321,   4.91472839,  -6.45656423,  -2.62612898,
        -8.51413128,   3.34151409,  12.88771229,  34.65904607,
       -13.35431265,   4.30981981,  21.88485042,   1.12385015,
        -2.91774249,  14.50057692,  -1.24198533,  -4.55757528,
        45.46247156,   8.26033485, -17.30334323,  -9.70654351,
       -11.24523311, -10.24578944,   2.52139723,   9.46442395,
        -2.6732785 ,  47.77493771,  -1.3560507 ,  -3.87657515,
        47.12672411,  19.07990287,  -0.9455647 ,   0.35427309,
         9.80390966,  19.42353943,   3.11970979,   3.44865065,
        21.42035593,  -7.88242488,   5.64544367,  10.20227577,
        -0.62195752,  14.75901715,   9.12831525,   7.21057155,
         4.53708356,  45.98559999,  -3.86594125,  50.79311745,
        25.99250789,  -3.46524068])

# |%%--%%| <ldwWMFrxzn|pVtWQEvZzp>
r"""°°°
Let's make a plot of the above samples
°°°"""
# |%%--%%| <pVtWQEvZzp|uXocRUtpfQ>

import matplotlib.pyplot as plt

plt.plot(feat, out, 'ro') # Plot the feature versus the output, using big red circles

# |%%--%%| <uXocRUtpfQ|paDQ3k8yKW>
r"""°°°
In order to use the linear gradient descent above to fit a polynomial, what we have to do is generate new features, from the existing ones. For example we could generate a new feature $feat_2$ and calculate its values as the square of $feat$: $feat_2 = feat^2$.

Similarly we could create more new features like:

$feat_3 = feat^3$

$feat_4 = feat^4$

$...$

Then we would have to normalise each of these new features, using its mean and standard deviation.
°°°"""
# |%%--%%| <paDQ3k8yKW|g7C2CKxPuG>
r"""°°°
To make our life easier, we will define a function that automates this process. It creates features up to a degree, calculates the means and standard deviations, and uses them to normalise them. The output would be the new design matrix for our problem.

Remember that once we calcualte our model, we should use the same procedure to generate and normalise features for new data points (our test set) before we feed them in the model. For this we would need to keep track of the means and standard deviations we used in the first place. Therefore, we can make our function return also these means and standard deviations so that we can keep them in a safe place. Finally, we can make our function use our pre-calculated means and standard deviations when we want to, instead of calculating new ones. So when we apply it on our test set, we can ask it to use the same means and standard deviations that we used for the training set.

You can skip this part and just use this function as is.
°°°"""
# |%%--%%| <g7C2CKxPuG|lZBZJZj6NG>

def mapFeatureAndNormalise_Polynomial(x, degree = 6, preCalcMeans = None, preCalcStd = None):
    '''
    Maps a single 1D feature to polynomial features up to the degree given
    Returns a new feature vector comprising of
    1, x, x^2, x^3, ..., x^degree
    '''
    
    x = x.reshape(-1, 1) #make into a vector if it has any other shape. The function size returns the number of elements in an array
    
    DesignMatrix = np.ones_like(x) # start with a column of ones
    
    for i in range(1, degree + 1):
            c = x ** i
            DesignMatrix = np.append(DesignMatrix, c, axis=1)

    # Calculate means and standard deviation if not provided
    if preCalcMeans is None:
        mu = DesignMatrix.mean(axis = 0)
    else:
        mu = preCalcMeans
        
    if preCalcStd is None:
        sigma = DesignMatrix.std(axis = 0, ddof=1)
    else:
        sigma = preCalcStd
    
    # Standardise
    for i in range(1, degree + 1):
        DesignMatrix[:,i] = (DesignMatrix[:,i] - mu[i]) / sigma[i]
                
    return DesignMatrix, mu, sigma

# |%%--%%| <lZBZJZj6NG|Lh2AM4nxLn>
r"""°°°
Let's create such new features up to the power of 2.
°°°"""
# |%%--%%| <Lh2AM4nxLn|LGcUP5WMNx>

X, mu, sigma = mapFeatureAndNormalise_Polynomial(feat, degree = 2)
print("Shape of design matrix: ", X.shape) # Expected shape = (# of samples, # features + 1)
print("Shape of means (and sigma): ", mu.shape) # Expected shape = # features + 1

# |%%--%%| <LGcUP5WMNx|FhvXCikQf5>
r"""°°°
Let's now use our gradient descent function to fit a 2-degree polynomial model to the same data. To do this, we should pass to the function the set of features up to the power of 2: `[1, feat, feat2]`
°°°"""
# |%%--%%| <FhvXCikQf5|qk6K4wArlX>

X, mu, sigma = mapFeatureAndNormalise_Polynomial(feat, degree = 2)

w, J = GradientDescent_np(X, out, alpha=0.1)

print('Estimated weights: ', w)  # Since we passed it three features (3 columns in our design matrix), we expect to receive three weight values

#Plot the evolution of the cost
plt.plot(np.arange(len(J)), J, 'bo')  # To plot the evolution of the Cost (J) we use as x a list of numbers from 1 to the number of elements inside J, and as y the values in J

# Plot line y = w0 + w1 * x + w2 * x^2
plt.figure() # We first create a new figure, otherwise it will include this plot in the previous one
plt.plot(feat, out, 'ro') # Here we plot the original feature vs out of our data as big red circles

# To plot our solution, we need to use the Thetas we have been given and calculate our estimated output for a series of points
xx = np.arange(feat.min(),feat.max(), 0.1) # Let's first define a series of points: from the minimum value of the original feature to the maximum value, every 0.1

# Now for each of our points in the range defined above, we need to calculate what our model gives us
# IMPORTANT: in order to pass the values in xx through our model, we need to create and normalise polynomial features in the same way as before!

yy = [w[0] + w[1] * (x-mu[1])/sigma[1] + w[2] * (x**2-mu[2])/w[2] for x in xx] # This is a pretty inefficient way to do this

plt.plot(xx, yy, 'g') # Now we can plot our points in the range vs the estimated value calculated by our model, connected by a green line

# |%%--%%| <qk6K4wArlX|zNVWdZ2xKG>
r"""°°°
Let's look in a more efficient way to pass new data through our model. We will create and normalise new features using the same `map_feature()` function, and the pre-calculated means and standard deviations. Then it should be just a matter of calculating a weighted sum (or else a dot product, vectorising this operation). It is important to see that

$y = w_0 + w_1 * x + w_2 * x^2$

is equivalent to 

$y = w_0 * x_0 + w_1 * x_1 + w_2 * x_2$, where $x_0 = 1$, $x_1 = x$ and $x_2 = x^2$

Importantly, doing this in this vectorised form, allows us to change the number of dimensions (degrees) without having to change our code.
°°°"""
# |%%--%%| <zNVWdZ2xKG|p6slsmHOC8>

# To plot our solution, we need to use the Thetas we have been given and calculate our estimated output for a series of points
xx = np.arange(feat.min(),feat.max(), 0.1) # Let's first define a series of points: from the minimum value of the original feature to the maximum value, every 0.1

X, mu, sigma = mapFeatureAndNormalise_Polynomial(xx, degree = 2, preCalcMeans = mu, preCalcStd = sigma)
yy = X @ w # @ is the operator for matrix multiplication. This is equivalent to yy = np.matmul(X, w)

plt.plot(feat, out, 'ro') # Here we plot the original feature vs out of our data as big red circles
plt.plot(xx, yy, 'g') # Now we can plot our points in the range vs the estimated value calculated by our model, connected by a green line
plt.show()

# |%%--%%| <p6slsmHOC8|yG6U9xaTCf>
r"""°°°
<font color=blue>What is the final cost value of the above fit?</font>
°°°"""
# |%%--%%| <yG6U9xaTCf|PBOmOZl494>

#Your Code Here#



# |%%--%%| <PBOmOZl494|CCP6Zh2v1p>
r"""°°°
<font color=blue>Can you fit a 3-degree polynomial and a 4-degree polynomial? What is the final cost value in these cases? HINT: to fit higher order polynomials you should just give more columns to your gradient descent, to include the corresponding 3-degree and 4-degree features.</font>
°°°"""
# |%%--%%| <CCP6Zh2v1p|cKgA2kIB5S>

#Your Code Here#



# |%%--%%| <cKgA2kIB5S|x0fpK2NRkx>
r"""°°°
<font color=blue> Can you reuse the above code to create the appropriate design matrices and try to fit polynomials of up to degree 15? How does the final cost change as you try polynomials of higher degrees?</font>
°°°"""
# |%%--%%| <x0fpK2NRkx|oI3P4Ngl4d>

#Your Code Here#


