# nyc_airbnb_data_science

**The Goal of Project**

The goal of the project is examining the data exploration and visualization techniques and building a machine learning models, Linear Regression and Neural Networks, to predict Airbnb prices. 

---

**Table of Contents**

* [Directory structure](#directory-structure)
* [Explaination of Project](#explaination-of-project)
* [Discussion of Results](#discussion-of-results)
* [About Author](#about-author)

---

## Directory structure

```
.
├── Data
│   ├── data.csv
│   ├── test_set_preprocessed.csv
│   └── train_set_preprocessed.csv
├── Airbnb_Exploration_and_Data_Analysis.ipynb
├── LICENSE
├── Prediction_of_Price.ipynb
├── README.md
└── requirements.txt
```

---

## Explaination of Project


The project consists of 4 main parts:
* Loading Data
* Data Wrangling
* Data Exploration and Analysis
* Price Prediction

The dataset is provided from [Kaggle Airbnb Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and because of that, it does not need too many data manupilation and cleaning process. One can reach the dataset with the [data.csv](https://github.com/singultek/nyc_airbnb_data_science/blob/main/Data/data.csv) folder.

Although dataset has already cleaned, it doesn't mean that one can directly take feed the machine learning models with the data. In terms of data wrangling, deleting unnecessary features, dealing with missing values can be done. Some of the features of dataset may not be useful to accomplish the prediction. In fact, some of them may decrease the model performance without giving any deep insight information of data. In order to prevent this situation, one can drop the name, id, host_name, last_review features. Even though last_revies could be used to enhance the performance of model, last_revies is dropped since that feature's 20 percent of values has been missing. Additionally, missing values of reviews_per_month is replaced by mean value of this feature. Replacing with mean value is selected over directly replace with zero due to the fact that these missing values won't affect significantly to the normalization process. 

In the data exploration process, exploring outlier is the focus point. Outlier needs close attention else it can result in wildly wrong estimations. Outlier is an observation that appears far away and diverges from an overall pattern in a sample. After exploring the data, minimum_nights and price are the top 2 features that outlier can cause wrong estimations. By eliminating the outliers in these 2 features, more generalized and accurate estimations can be achieved. 

Another key tools of the data analysis is of course visualization. By the help of visualization, one can understand better the data, behaviour and distribution of data and relation between features. One can stress out the most inportant outcome as the need of converting price feature to logarithmic scale. Coversion of log scale gives better scaled values and distribution of log-price resembles the Gaussian distribution.

Before getting into the price prediction, another dramatically effective steps will be performed. These steps are encoding some features, performing normalization and spliting data to training and test datasets. Encoding is necesary for non-integer values. One need to encode the neighbourhood_group, neighbourhood and room_type features. Then, splitting the dataset is succeed and saved into 2 different csv files, [train_set_preprocessed.csv](https://github.com/singultek/nyc_airbnb_data_science/blob/main/Data/train_set_preprocessed.csv) and [test_set_preprocessed.csv](https://github.com/singultek/nyc_airbnb_data_science/blob/main/Data/test_set_preprocessed.csv). Finally, normalization process is performed to the train and test dataset with the input features. The target portion of the train and test datasets won't be normalized. On the other hand, normalizing the target datasets and after predictions getting back to normal scale could be another solution aspect to the problem.

After completing the preprocessing steps, estimation of the price can be computed. 2 different model will be used, Linear Regression and Neural Networks. Inside the Linear Regression models, additional to pure linear regression, couple of alternatives with regularization terms will be tested. In order to find the best hyperparameters, Grid Search technique is used. As a second model, basic neural network model is built with the help of tensorflow.

---

### Discussion of Results

| model                               |     MAE     |     RMSE     |    R2 SCORE   |
|-------------------------------------|-------------|--------------|---------------|
| Linear Regression                   |   0.3755    |    0.4999    |     0.5245    |
| Linear Regression with Lasso        |   0.3753    |    0.4995    |     0.5254    |
| Linear Regression with Ridge        |   0.3755    |    0.4999    |     0.5246    |
| Linear Regression with ElasticNet   |   0.3754    |    0.4997    |     0.5249    |
| Neural Network                      |   0.3907    |    0.5114    |     0.4966    |

3 metrics is based on the measuring the goodness of the predictions. Mean Absolute Error (MAE) is the distance between predictions and actual values. Root Mean Square Error (RMSE) that is the root of MSE (Mean Square Error). MSE is the distance between predictions and actual values. It shows how accurately the model predicts the response. R2 will be calculated to find the goodness of fit measure. 

One can see from the results, the overall performances of the both models are not promising. Roughly, one can state that model represents the data with 50%, which can be considered barely as decent. Additional outcome of the table is that regularization terms doesn't help as much as expected. The explanation of this can be nearly underfitting model. Regularization is helpful technique to prevent the overfitting. In this case, both models are pretty far away from the overfitting and due to that reason, regularization is not affecting the performance of Linear Regression.

---


### About Author

I'm Sinan Gültekin, a master student in Computer and Automation Engineering at the University of Siena. 

For any suggestions or questions, you can contact me via <singultek@gmail.com>

Distributed under the GPL-3.0 License. _See ``LICENSE`` for more information._
