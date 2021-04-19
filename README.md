# Improving National Financial Well-Being Through Machine Learning and Artificial Intelligence

![image](https://therightnewsnetwork.com/wp-content/uploads/2019/03/TRNN-Revelations-Finance.png)

Table of Contents :bookmark_tabs:
=================
- [Executive Summary](#executive-summary)
- [Introduction](#introduction)
- [Analytics Approach](#analytics-approach)
- [Data Preparation and Understanding](#data-preparation-and-understanding)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Proposed Business Recommendations](#proposed-business-recommendations)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Code and Resources Used](#code-and-resources-used)

## Executive Summary
This report aims to identify solutions and business recommendations to help the Ministry of Social and Family Development (MSF) better identify people with poor financial well-being which tends to lead to unhappiness. 

Singapore is reputed for its high Gross Domestic Product (GDP) per capita, life expectancy, and integrity. Yet, it has a relatively low happiness level at 31st ranking in the world. To improve happiness in Singapore, our team has chosen to target an aspect that the Singapore Government can likely influence and intervene in. The aspect is financial well-being in which freedom to make life choices is a significant factor.

We adopted a data-driven approach to analyze the personal data that the government collects to aggregate a more holistic judgment of one’s financial well-being through the usage of machine learning and artificial intelligence techniques. Data preprocessing steps such as data cleaning, discretization of the target variable (“At risk” and “Not at risk”), and feature selection were performed followed by data transformation before feeding into the machine learning models. Following that, we developed 7 machine learning models for binary classification: logistic regression, decision tree classifier, random forest classifier, XGBoost classifier, support vector classifier, Bernoulli naïve Bayes classifier, and neural network. A 70:30 train-test split was used to partition the data set for evaluation purposes. Hyperparameter tuning was performed on the training data set to extract the optimal performance out of the models.

The 7 machine learning models were evaluated using various evaluation metrics such as accuracy, precision, recall, and F1-Score. F1-Score was chosen as the final criteria for model selection to account for imbalanced data and the cost of misclassification. All the models managed to beat the baseline F1-Score which was calculated to be at 0.3333. Support vector classifier emerged as the winner providing the highest F1-Score at 0.7194. As such, the support vector classifier was chosen to be recommended to MSF. 

A few business recommendations were proposed to utilize the machine learning model. Firstly, ideas were developed to facilitate the application of the proof of concept to the actual dataset and identify the target group to assist. Secondly, to simplify the workflow for MSF’s transition into interacting with our machine learning model, a graphical user interface was proposed. They can directly export the collected data into the system and the candidates who belong to the “At risk” group along with the relevant information will be generated automatically. Thirdly, a longitudinal study can be conducted to evaluate the efficacy of the intervention by MSF.

The limitations of the data and models were discussed. Limitations of models include the tradeoff between interpretability and predictability and the difficulty in predicting social and dynamic systems. Suggested future work includes survey questions reduction using a randomized approach to reduce survey fatigue, change of target variable to happiness or health scores, and shift of focus from freedom to make life’s choices to social support and generosity, and the other factors which performed poorly when used to determine the Happiness Index in Singapore. 

## Introduction


## Analytics Approach
Today, government agencies have access to a great deal of personal data (both financial and non-financial data) of their citizens, such as phone and traffic records, health records (including genetic records), spending habits, water and electricity consumption, and even online browsing activities.

In Singapore, citizens receive financial aid from various Ministries (MSF, MOM, MOE) according to their per capita income as well as housing type. Yet, we know that financial well-being is unable to be captured fully through basic financial measures such as income or housing. This grey area requires a more sophisticated solution through a data-driven approach.

Using machine learning and artificial intelligence techniques, we aim to leverage data analytics to sieve through the wide array of personal data that the government collects to aggregate a better and more holistic judgment of a person’s financial well-being and hence extend financial help to those struggling to obtain financial well-being and happiness.

## Data Preparation and Understanding
### Data Set Description
To ensure that our model design would be able to cater to the types of citizen profile data that would be used in real-life implementation, the dataset selected for this project was one that the team deemed to have sufficient nexus with the actual data that is likely to be available. The main criteria for a suitable dataset were that it needed to have a qualifying variable to determine financial well-being, as well as a variety of other variables commonly found in government databases. 

The dataset we have selected is from a survey conducted by the Consumer Financial Protection Bureau (CFPB) in 2015. The dataset consists of 217 columns and 6395 rows. The full list of variables is provided in Appendix A. The column “FWBscore” is a discrete variable that indicates a person’s financial well-being on a scale of 0-100. 

<p align="center">
  “Financial Well-Being Survey Data” from Kaggle:
  https://www.kaggle.com/anthonyku1031/nfwbs-puf-2016-data 
</p>

The “FWBscore” scale was determined via an extensive research study involving cognitive interviews, factor testing, and psychometric testing to accurately measure a person’s financial well-being according to four key elements: (i) control over day-to-day finances, (ii) capacity to absorb financial shock, (iii) financial freedom to make choices to enjoy life, and (iv) being on track to meet financial goals. The result was a 10 question scale-based scoring system that accorded each person in the dataset a holistic and comprehensive indicator of their financial well-being. Further elaboration on the survey and methods can be found in Appendix B.

In addition to asking the 10 questions directly related to computing the FWB Score, indirect questions such as financial knowledge, education level, income and employment, family history, financial habits, demographic information were also asked in the survey. 

This dataset provides us with a multitude of personal data available to a government agency as well as a highly accurate proxy for actual financial well-being “FWBscore”. This would allow our team to develop predictive models and test the accuracy of utilizing personal data to predict financial well-being.




## Model Development and Evaluation
![image](https://user-images.githubusercontent.com/45563371/115279086-d49ae300-a178-11eb-9a21-1ff63601f805.png)

## Proposed Business Recommendations

## Limitations

## Future Work

## Conclusion
In conclusion, the machine learning model facilitates the data-driven approach to identify people in Singapore who might experience higher levels of unhappiness due to poor financial well-being. We accomplish this by first obtaining a complete and cleaned dataset through various data preprocessing steps. The processed and transformed data was fed into the machine learning models. The support vector classifier performed the best based on F1-Score and was recommended as the model of choice for the Ministry of Social and Family Development (MSF). 

Through the data-driven approach, our solution overcomes human biases and provides accurate and timely information for better decision-making. In the short and medium term, the machine learning model would help MSF to better identify citizens who have poorer financial well-being which leads to unhappiness, and provide them aid. Fine-tuning of policies serves as a long-term measure to improve the happiness index of Singapore. 

## Code and Resources Used
**Python:** Version 3.7.10

**Packages:** pandas, numpy, matplotlib, seaborn, category_encoders, tensorflow, xgboost, keras, sklearn

**Dataset:** https://www.kaggle.com/anthonyku1031/nfwbs-puf-2016-data
