# nyc_airbnb_data_science

**The Goal of Project**
The goal of the project is examining the data exploration and visualization techniques and building a machine learning models, Linear Regression and Neural Networks, to predic Airbnb prices. 

---
**Table of Contents**

* [Directory structure](#directory-structure)
* [Explaination of project](#explaination-of-project)
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
├── Prediction_of_Price.ipynb
├── LICENSE
├── README.md
├── requirements.txt
```

---
## Explaination of project

The project consists of 4 main part:
* Loading Data
* Data Wrangling
* Data Exploration and Analysis
* Price Prediction

The dataset is provided from [Kaggle Airbnb Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) and because of that, it does not need too many data manupilation and cleaning process. One can reach the dataset in the [Data](https://github.com/singultek/nyc_airbnb_data_science/blob/main/Data/data.csv) folder.
Although dataset has already cleaned, it doesn't mean that one can directly take feed the machine learning models with the data. In terms of data wrangling, deleting unnecessary features, dealing with missing values can be done. Some of the features of dataset may not be useful to accomplish the prediction. In fact, some of them may decrease the model performance without giving any deep insight information of data. In order to prevent this situation, one can drop the name, id, host_name, last_review features. Even though last_revies could be used to enhance the performance of model, last_revies is dropped since that feature's 20 percent of values has been missing. Additionally, missing values of reviews_per_month is replaced by mean value of this feature. Replacing with mean value is selected over directly replace with zero due to the fact that these missing values won't affect significantly to the normalization process. 



---

### Discussion of Results

| model                               |     MAE     |     RMSE     |    R2 SCORE   |
|-------------------------------------|-------------|--------------|---------------|
| Linear Regression                   |   0.3755    |    0.4999    |     0.5245    |
| Linear Regression with Lasso        |   0.3753    |    0.4995    |     0.5254    |
| Linear Regression with Ridge        |   0.3755    |    0.4999    |     0.5246    |
| Linear Regression with ElasticNet   |   0.3754    |    0.4997    |     0.5249    |
| Neural Network                      |   0.3907    |    0.5114    |     0.4966    |

```
  



### About Author

I'm Sinan Gültekin, a master student on Computer and Automation Engineering at University of Siena. 

For any suggestions or questions, you can contact me via <singultek@gmail.com>

Distributed under the GPL-3.0 License. _See ``LICENSE`` for more information._
