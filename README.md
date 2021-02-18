# Machine Learning Fundamentals - Cumulative Lab

## Introduction

In this cumulative lab, you will work through an end-to-end machine learning workflow, focusing on the fundamental concepts of machine learning theory and processes. The main emphasis is on modeling theory (not EDA or preprocessing), so we will skip over some of the data visualization and data preparation steps that you would take in an actual modeling process.

## Objectives

You will be able to:

* Recall the purpose of a train-test split
* Practice performing a train-test split
* Recall the difference between bias and variance
* Practice identifying bias and variance in model performance
* Practice applying strategies to minimize bias and variance
* Practice selecting a final model and evaluating it on a holdout set

## Your Task: Build a Model to Predict Blood Pressure

![stethoscope sitting on a case](images/stethoscope.jpg)

<span>Photo by <a href="https://unsplash.com/@marceloleal80?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Marcelo Leal</a> on <a href="https://unsplash.com/s/photos/blood-pressure?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Business and Data Understanding

Hypertension (high blood pressure) is a treatable condition, but measuring blood pressure requires specialized equipment that most people do not have at home.

The question, then, is ***can we predict blood pressure using just a scale and a tape measure***? These measuring tools, which individuals are more likely to have at home, might be able to flag individuals with an increased risk of hypertension.

[Researchers in Brazil](https://doi.org/10.1155/2014/637635) collected data from several hundred college students in order to answer this question. We will be specifically using the data they collected from female students.

The measurements we have are:

* Age (age in years)
* BMI (body mass index, a ratio of weight to height)
* WC (waist circumference in centimeters)
* HC (hip circumference in centimeters)
* WHR (waist-hip ratio)
* SBP (systolic blood pressure)

The chart below describes various blood pressure values:

<a title="Ian Furst, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Hypertension_ranges_chart.png"><img width="512" alt="Hypertension ranges chart" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Hypertension_ranges_chart.png/512px-Hypertension_ranges_chart.png"></a>

### Requirements

#### 1. Perform a Train-Test Split

Load the data into a dataframe using pandas, separate the features (`X`) from the target (`y`), and use the `train_test_split` function to separate data into training and test sets.

#### 2. Build and Evaluate a First Simple Model

Using the `LinearRegression` model and `mean_squared_error` function from scikit-learn, build and evaluate a simple linear regression model using the training data. Also, use `cross_val_score` to simulate unseen data, without actually using the holdout test set.

#### 3. Use `PolynomialFeatures` to Reduce Underfitting

#### 4. Use Regularization to Reduce Overfitting

#### 5. Evaluate a Final Model on the Test Set

## 1. Perform a Train-Test Split

Before looking at the text below, try to remember: why is a train-test split the *first* step in a machine learning process?

.

.

.

A machine learning (predictive) workflow fundamentally emphasizes creating *a model that will perform well on unseen data*. We will hold out a subset of our original data as the "test" set that will stand in for truly unseen data that the model will encounter in the future.

We make this separation as the first step for two reasons:

1. Most importantly, we are avoiding *leakage* of information from the test set into the training set. Leakage can lead to inflated metrics, since the model has information about the "unseen" data that it won't have about real unseen data. This is why we always want to fit our transformers and models on the training data only, not the full dataset.
2. Also, we want to make sure the code we have written will actually work on unseen data. If we are able to transform our test data and evaluate it with our final model, that's a good sign that the same process will work for future data as well.

### Loading the Data

In the cell below, we import the pandas library and open the full dataset for you. It has already been formatted and subsetted down to the relevant columns.


```python
# Run this cell without changes
import pandas as pd
df = pd.read_csv("data/blood_pressure.csv", index_col=0)
df
```


```python
# __SOLUTION__
import pandas as pd
df = pd.read_csv("data/blood_pressure.csv", index_col=0)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>bmi</th>
      <th>wc</th>
      <th>hc</th>
      <th>whr</th>
      <th>SBP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>28.76</td>
      <td>88</td>
      <td>101</td>
      <td>87</td>
      <td>128.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>27.59</td>
      <td>86</td>
      <td>110</td>
      <td>78</td>
      <td>123.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>22.45</td>
      <td>72</td>
      <td>104</td>
      <td>69</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>28.16</td>
      <td>89</td>
      <td>108</td>
      <td>82</td>
      <td>126.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>25.05</td>
      <td>81</td>
      <td>108</td>
      <td>75</td>
      <td>120.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>219</th>
      <td>21</td>
      <td>45.15</td>
      <td>112</td>
      <td>132</td>
      <td>85</td>
      <td>157.00</td>
    </tr>
    <tr>
      <th>220</th>
      <td>24</td>
      <td>37.89</td>
      <td>96</td>
      <td>124</td>
      <td>77</td>
      <td>124.67</td>
    </tr>
    <tr>
      <th>221</th>
      <td>37</td>
      <td>33.24</td>
      <td>104</td>
      <td>108</td>
      <td>96</td>
      <td>126.67</td>
    </tr>
    <tr>
      <th>222</th>
      <td>28</td>
      <td>35.68</td>
      <td>103</td>
      <td>130</td>
      <td>79</td>
      <td>114.67</td>
    </tr>
    <tr>
      <th>223</th>
      <td>18</td>
      <td>36.24</td>
      <td>113</td>
      <td>128</td>
      <td>88</td>
      <td>119.67</td>
    </tr>
  </tbody>
</table>
<p>224 rows × 6 columns</p>
</div>



### Identifying Features and Target

Once the data is loaded into a pandas dataframe, the next step is identifying which columns represent features and which column represents the target.

Recall that in this instance, we are trying to predict systolic blood pressure.

In the cell below, assign `X` to be the features and `y` to be the target. Remember that `X` should **NOT** contain the target.


```python
# Replace None with appropriate code

X = None
y = None

X
```


```python
# __SOLUTION__
X = df.drop("SBP", axis=1)
y = df["SBP"]

X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>bmi</th>
      <th>wc</th>
      <th>hc</th>
      <th>whr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31</td>
      <td>28.76</td>
      <td>88</td>
      <td>101</td>
      <td>87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>27.59</td>
      <td>86</td>
      <td>110</td>
      <td>78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>22.45</td>
      <td>72</td>
      <td>104</td>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>28.16</td>
      <td>89</td>
      <td>108</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>25.05</td>
      <td>81</td>
      <td>108</td>
      <td>75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>219</th>
      <td>21</td>
      <td>45.15</td>
      <td>112</td>
      <td>132</td>
      <td>85</td>
    </tr>
    <tr>
      <th>220</th>
      <td>24</td>
      <td>37.89</td>
      <td>96</td>
      <td>124</td>
      <td>77</td>
    </tr>
    <tr>
      <th>221</th>
      <td>37</td>
      <td>33.24</td>
      <td>104</td>
      <td>108</td>
      <td>96</td>
    </tr>
    <tr>
      <th>222</th>
      <td>28</td>
      <td>35.68</td>
      <td>103</td>
      <td>130</td>
      <td>79</td>
    </tr>
    <tr>
      <th>223</th>
      <td>18</td>
      <td>36.24</td>
      <td>113</td>
      <td>128</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
<p>224 rows × 5 columns</p>
</div>



Make sure the assert statements pass before moving on to the next step:


```python
# Run this cell without changes

# X should be a 2D matrix with 224 rows and 5 columns
assert X.shape == (224, 5)

# y should be a 1D array with 224 values
assert y.shape == (224,)
```


```python
# __SOLUTION__

# X should be a 2D matrix with 224 rows and 5 columns
assert X.shape == (224, 5)

# y should be a 1D array with 224 values
assert y.shape == (224,)
```

### Performing Train-Test Split

In the cell below, import `train_test_split` from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)).

Then create variables `X_train`, `X_test`, `y_train`, and `y_test` using `train_test_split` with `X`, `y`, and `random_state=42`.


```python
# Replace None with appropriate code

# Import the relevant function
None

# Create train and test data using random_state=42
None, None, None, None = None
```


```python
# __SOLUTION__

# Import the relevant function
from sklearn.model_selection import train_test_split

# Create train and test data using random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Make sure that the assert statements pass:


```python
# Run this cell without changes

assert X_train.shape == (168, 5)
assert X_test.shape == (56, 5)

assert y_train.shape == (168,)
assert y_test.shape == (56,)
```


```python
# __SOLUTION__

assert X_train.shape == (168, 5)
assert X_test.shape == (56, 5)

assert y_train.shape == (168,)
assert y_test.shape == (56,)
```

## 2. Build and Evaluate a First Simple Model

For our baseline model (FSM), we'll use a `LinearRegression` from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

### Instantiating the Model

In the cell below, instantiate a `LinearRegression` model and assign it to the variable `baseline_model`.


```python
# Replace None with appropriate code

# Import the relevant class
None

# Instantiate a linear regression model
baseline_model = None
```


```python
# __SOLUTION__

# Import the relevant class
from sklearn.linear_model import LinearRegression

# Instantiate a linear regression model
baseline_model = LinearRegression()
```

Make sure the assert passes:


```python
# Run this cell without changes

# baseline_model should be a linear regression model
assert type(baseline_model) == LinearRegression
```


```python
# __SOLUTION__

# baseline_model should be a linear regression model
assert type(baseline_model) == LinearRegression
```

If you are getting the type of `baseline_model` as `abc.ABCMeta`, make sure you actually invoked the constructor of the linear regression class with `()`.

If you are getting `NameError: name 'LinearRegression' is not defined`, make sure you have the correct import statement.

### Fitting and Evaluating the Model on the Full Training Set

In the cell below, fit the model on `X_train` and `y_train`:


```python
# Your code here
```


```python
# __SOLUTION__
baseline_model.fit(X_train, y_train)
```




    LinearRegression()



Then, evaluate the model using root mean squared error (RMSE). To do this, first import the `mean_squared_error` function from scikit-learn ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)). Then pass in both the actual and predicted y values, along with `squared=False` (to get the RMSE rather than MSE).


```python
# Replace None with appropriate code

# Import the relevant function
None

# Generate predictions using baseline_model and X_train
y_pred_baseline = None

# Evaluate using mean_squared_error with squared=False
baseline_rmse = None
baseline_rmse
```


```python
# __SOLUTION__

# Import the relevant function
from sklearn.metrics import mean_squared_error

# Generate predictions using baseline_model and X_train
y_pred_baseline = baseline_model.predict(X_train)

# Evaluate using mean_squared_error with squared=False
baseline_rmse = mean_squared_error(y_train, y_pred_baseline, squared=False)
baseline_rmse
```




    13.404369445571641



Your RMSE calculation should be around 13.4:


```python
# Run this cell without changes
assert round(baseline_rmse, 1) == 13.4
```


```python
# __SOLUTION__
assert round(baseline_rmse, 1) == 13.4
```

This means that on the *training* data, our predictions are off by about 13 mmHg on average.

But what about on *unseen* data?

To stand in for true unseen data (and avoid making decisions based on this particular data split, therefore not using `X_test` or `y_test` yet), let's use cross-validation.

### Fitting and Evaluating the Model with Cross Validation

In the cell below, import `cross_val_score` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)) and call it with `baseline_model`, `X_train`, and `y_train`.

For specific implementation reasons within the scikit-learn library, you'll need to use `scoring="neg_root_mean_squared_error"`, which returns the RMSE values with their signs flipped to negative. Then we take the average and negate it at the end, so the number is directly comparable to the RMSE number above.


```python
# Replace None with appropriate code

# Import the relevant function
None

# Get the cross validated scores for our baseline model
baseline_cv = None

# Display the average of the cross-validated scores
baseline_cv_rmse = -(baseline_cv.mean())
baseline_cv_rmse
```


```python
# __SOLUTION__

# Import the relevant function
from sklearn.model_selection import cross_val_score

# Get the cross validated scores for our baseline model
baseline_cv = cross_val_score(baseline_model, X_train, y_train, scoring="neg_root_mean_squared_error")

# Display the average of the cross-validated scores
baseline_cv_rmse = -(baseline_cv.mean())
baseline_cv_rmse
```




    13.797218918749715



The averaged RMSE for the cross-validated scores should be around 13.8:


```python
# Run this cell without changes

assert round(baseline_cv_rmse, 1) == 13.8
```


```python
# __SOLUTION__

assert round(baseline_cv_rmse, 1) == 13.8
```

### Analysis of Baseline Model

So, we got about 13.4 RMSE for the training data, 13.8 RMSE for the test data. RMSE is a form of *error*, so this means the performance is somewhat better on the training data than the test data.

Referring back to the chart above, both errors mean that on average we would expect to mix up someone with stage 1 vs. stage 2 hypertension, but not someone with normal blood pressure vs. critical hypertension. So it appears that the features we have might be predictive enough to be useful.

Are we overfitting? Underfitting?

.

.

.

The RMSE values for the training data and test data are fairly close to each other, so we are probably not overfitting too much, if at all.

It seems like our model has some room for improvement, but without further investigation it's impossible to know whether we are underfitting, or we are simply missing the features that we would need to make predictions with less error. (For example, we don't know anything about the diets of these study participants, and we know that diet can influence blood pressure.)

In the next step, we'll attempt to reduce underfitting by applying some polynomial features transformations to the data.
