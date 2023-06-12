# Predict Housing Prices Using Advanced Regression Models
In this project I explored the Boston Housing Prices dataset from Kaggle [1] using advanced regression models. We 
perform various forms of feature extraction and data visualization to first understand the dataset. We then further
build on this work and considered the following model architectures: Linear Regression, SVR, Random Forest, XGBoost, 
GPR, and Artificial Neural Network. This work highlights the methods that can be in place for working with tabular 
datasets. This format can easily be adapted for other datasets or problem types (e.g. classification) as the steps are
well documented. Additionally, we highlight several additional techniques that can be used with larger datasets.
In general, given more complex problem domains, we can build customized solutions using more advanced methods. For
example, custom GPR Kernels using Gpy are available and can provide major benefits given the right implementation.


### Table of Contents
1. Dataset Exploration
2. Feature Selection
3. Modeling
4. Results
5. Understanding the Git Repo
6. References

### 1. Dataset Exploration

The first step in any machine learning project is to understand the data we are working with. We begin by first
investigating the statistics of our dataset to see the types of features we are dealing with and the ranges
we have. This gives us intuition on the type of model and data processing required. 

From [1] we see there are 13
variables of interest (columns 1-13) and one target variable we hope to predict (MEDV). The variables are defined as 
follows:  

    Inputs Features:
    CRIM: per capita crime rate by town
    ZN: proportion of residential land zoned for lots over 25,000 sq. ft.
    INDUS: proportion of non-retail buisness acres per town
    CHAS: Charles River dummy Variable (= 1 if tract bounds river; 0 otherwise)
    NOX: nitric oxides concentration (parts per 10 million)
    RM: averge number of rooms per dwelling
    AGE: proportion of owner-occupied units built prior to 1940
    DIS: weighted distances to five Boston employment centers
    RAD: index of accessibility to radial highways
    TAX: full-value property tax rate per $10K
    PTRATIO: pupil-teacher ratio by town
    B: 1000(Bk-0.63)2 where Bk is the proportion of blacks by town
    LSTAT: pct lower status of the population

    Output Feature:
    MEDV: medivan value of owner-occupied homes in $1000s

From Table 1 below we see basic statistics for each of our variables. In particular, we see there are no null values that
need to be adjusted for. It is clear that the mean of each variable is very different and thus some form of preprocessing
is preferred to allow for equal learning from featured inputs. Though we do not know the actual distribution of each of our 
featured inputs, looking at the data it seems that the standard normal distribution may not be the best assumption.
For example, the ZN variable has most our variance in the highest quartile of our dataset. Thus, we instead recommend
using standard 0-1 scaling for all our variables as the preprocessing. 

![](images/Dataset_Statistics.png)

From Figure 1A below we can see the various linear relationships between our variables and the corresponding 
pearson r correlation coefficient. From here we see that two of our featured inputs have a very strong linear relationship
with our target variable: LSTAT (p=-0.74) and RM (p=0.7). The rest of our variables seem to have p values less than 
0.5 with weaker relationships. As a result, we don't expect our linear model to work off the bat. More complex relationships
need to be determined from other model structures. From Figure 1B, we can see the relationship between all our variables.
One important find from Figure 1B is that the DIS variable is highly correlated to several of our other variables such as 
NDUS, NOX, and AGE. For now we will keep this variable; however it may be worth further modeling to see if removing this variable
can reduce model complexity without impacting our performance. From Figure 1C we can infer that over half our variance can be
visualized using only our first principal component. And we can obtain 95% of our variance using only 8 components.

![](images/Dataset_Exploration.png)


### 2. Feature Selection

We implemented various feature selection methods to get a better understanding of our dataset.
Looking first at the pearson r correlation matrix in Figure 1B, we see the highest correlations between
our existing variables exist between RAD-TAX (p=0.91), NOX-DIS (p=-0.77), INDUS-NOX (p=0.77) and 
AGE-DIS (p=-0.76). Furthermore, from Figure 1C, we can see that with only 8 principal components, 
we can derive >95% of the variance in the data.

We further implement Univariate Feature Selection, Recursive Feature Elimination, and Tree Based
Feature Selection using Sklearn. In particular we implement Univariate Feature Selection using F Regression,
R Regression, and Mutual Information. RFE is implemented using an SVR Linear Kernel. Tree Based Feature
Selection is implemented using an ExtraTreesRegressor. The results are seen in Table 2.

From Table 2 we can see that there is a general concensus that LSTAT, RAD, PRRATIO and TAX are
the most important features in predicting MEDV. This is something we can keep in mind as we continue to model.
An interesting controlled experiment would be see the impact of modeling using only these 4 variables as 
feature inputs. 

![](images/Feature_Selection.png)


### 3. Modeling

We implement 6 different models for predicting the MEDV value. Note that our dataset is relatively small
with only 506 training points (Table 1), each of dimension 13. As a result, we have to limit our model 
complexity to this domain. Thus, we implement a regularized linear regression model (Elastic Net), 
Support Vector Regression (SVR), Random Forest, XGBoost, Gaussian Process Regression, and a simplistic
Artificial Neural Network (ANN). 

For each model architecture except the ANN we perform an extensive hyper-parameter search using
4-fold cross validation and grid search. The best model is selected using negative RMSE. 
Metrics are reported in RMSE and %Error for each model. Plots showing the observed vs. predicted
value are shown for each model.

There are additional model architectures that have shown SOTA results in tabular regression tasks
such as TabNet. These methods are not implemented here but can be considered for future iterations.


### 4. Results

The results from our models can be seen in the table below. We focus on RMSE as our key metric for assessment
as the percent error can be biased based on the size of the values in the target variable. 

From this table we see that XGBoost has the best performance on our test set. Note
our test set was randomly selected as 20% of the entire dataset. The split was held the same for all
our models.

|                       | **Train RMSE** | **Train %Error** | **Test RMSE** | **Test %Error** |
|-----------------------|:--------------:|:----------------:|:-------------:|:---------------:|
| **Linear Regression** |      4.65      |       16.5       |      4.91     |       16.8      |
| **SVR**               |      2.68      |        7.4       |      3.31     |       9.9      |
| **Random Forest**     |      1.46      |        4.9       |      3.19     |       10.3      |
| **XGBoost**           |      0.55      |        2.1       |      2.63     |       9.9       |
| **GPR**               |      1.64      |        6.2       |      3.28     |       10.8      |
| **ANN**               |      2.52      |       10.0       |      3.19     |       12.3      |

Figure 2 below shows the predicted vs. observed value for MEDV for each of our trained models.
This allows us to see the range in which our model was performing better or worse. Note how
all our models tend to get worse as we get to higher values of MEDV. There must either be limited data
for learning in this spectrum or we more complex relationships which were not captured by the models.

![](images/Model_Predictions.png)


### 5. Understanding this Git Repo
There are three main parts to the repo.
1. The main.py file is our centralized function that handles which models we end up running or the order of
evaluations.
   
2. The *models* folder is where we store each of our individual model implemenations. Details of hyper-parameter
tuning to plotting conditions can all be found here. Inside this folder we have a model_helper_fxns.py file which
   holds all our centralized functions that are used in all models.
   
3. The *dataset* folder is where we store both the dataset as well as the dataset preprocessing functions. In particular
we handle the dataset preprocessing (scaling, loading) and feature extraction in this area.

### 6. References  
[1] https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data?resource=download  
[2] https://scikit-learn.org/stable/modules/feature_selection.html
