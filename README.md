# SC1015: Data Science Mini Project - Predicting the Nasdaq100: A Comparative Analysis using ARIMA and LSTM Models


Lab Group: `C126`

Team: `6`

Members: 

1. Tan Liang Meng

2. Hui Xujia 

3. Lim Ricky 

- - -

## Description:

This GitHub repository serves as a comprehensive resource for the Mini Project in SC1015: Introduction to Data Science and AI. Included are Jupyter Notebooks, datasets, images, video presentations, and source materials/references utilized and created throughout the project.

The Jupyter Notebooks provide detailed descriptions and smaller details not covered in this README. To facilitate access and organization, the notebooks are divided into five parts corresponding to the project's main sections.

## File Arrangement

```
.
├── datasets
│   └── Nasdaq100.csv
├── Models-Final.ipynb
├── README.md
└── requirements.txt
```

- - -

## Table of contents:
1. [Overview](#1-Overview-open_book) 📖

2. [Problem Statement](#2-Problem-Statement-brain) :brain:

3. [Data Preparation and Cleaning](#3-Data-Preparation-and-Cleaning-broom) :broom: 

4. [Exploratory Data Analysis](#4-Exploratory-Data-Analysis-rocket) :rocket: 

5. [Models](#5-Models-robot) :robot: 

6. [Data Driven Insights and Conclusion](#6-Data-Driven-Insights-and-Conclusion-left_speech_bubble) :left_speech_bubble: 

7. [Contributors](#7-Contributors-black_nib) :black_nib:

8. [References](#8-References-page_facing_up) :page_facing_up:

- - -
## 1. Overview 📖
This project aims to predict the trend of the Nasdaq100 index based on its components. This project uses two models, ARIMA(Autoregressive Integrated Moving Average) and LSTM (Long Short-Term Memory networks), to predict the future values of the index.<br><br>
Historical data on stocks, indexes, and their prices was obtained by manually downloading a dataset from Investing.com's financial markets platform.
## 2. Problem Statement :brain:

Data set used: https://www.investing.com/indices/nq-100
### Problem Definition
- Which index is more suited for predicting market trend?
	- Can a Model be used to predict the market trend?
	- Which is the best Model to be used for predicting market trends?

## 3. Data Preparation and Cleaning :broom:
In this section, we performed a thorough data cleaning and preparation process to enhance the quality of our dataset. Our objective was to ensure that the data is suitable for use in machine learning models in subsequent sections.

Key steps taken:
- Sorting the data from earliest to the latest
- Dropping unwanted columns, focus only on nasdaq100
- Splitting the time-series into train and test set before transformation - to avoid data leakage
- Data Transformation - Differencing improve stationarity (For ARIMA model)
- Data Normalization using MinMaxScale (For LSTM model)
- The code of Data Transformation and Data Normalization is in the model part not in the data preparation part.

## 4. Exploratory Data Analysis :rocket:
- Using the 6 Days moving average, we can see that It is not able to predict a significant jump or drop in price.<br>
- Using time series plot, ACF, PACF plot we see that the time series is not  stationary
- Using the same method, we see that the first-differenced time series is stationary
- From ACF, PACF and the stationarity of the time series after first order differencing, we decided to choose the parameter (1,1,1) for the ARIMA model

## 5. Models :robot:
### ARIMA
In our exploratory data analysis, we discovered that the Nasdaq100 time series is non-stationary. We decided to use the ARIMA model to forecast the future values of the index due to the following reasons 
- ARIMA models are **flexible and adaptable** to many types of time series data, including non-stationary data.
- ARIMA models **improve forecasting accuracy** by addressing non-stationarity through differencing.

### LSTM
We realized that the ARIMA model was unable to capture the complex non-linear trend. To better capture the complex non-linear trends present in the data, we then explored the LSTM model due to the following reasons:
- LSTM model is designed to **capture non-linear relationships**
- LSTM uses memory cells to **store information** about past inputs and outputs, allowing them to capture long-term dependencies in the data. 



## 6. Conclusion :left_speech_bubble:
### ARIMA Model
- Able to capture the general range of the value
- Trend was not accurately captured

### LSTM Model
- Performed better than the ARIMA model in capturing the complex non-linear trends
- Accurately predicting future values. 


## 7. Contributors :black_nib:

Hui Xujia:		LSTM Model, 

Tan Liang Meng:		ARIMA Model,

Lim Ricky:		Data Preparation, Documentation,
  
- - -  
## 8. References :page_facing_up:
[https://analyticsindiamag.com/quick-way-to-find-p-d-and-q-values-for-arima/](https://analyticsindiamag.com/quick-way-to-find-p-d-and-q-values-for-arima/)<br>
[https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7](https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7)<br>
[https://zhuanlan.zhihu.com/p/266994943](https://zhuanlan.zhihu.com/p/266994943)<br>
[https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python](https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python)<br>
[https://keras.io/api/layers/recurrent_layers/lstm/#lstm-class](https://keras.io/api/layers/recurrent_layers/lstm/#lstm-class)<br>
[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)<br>
