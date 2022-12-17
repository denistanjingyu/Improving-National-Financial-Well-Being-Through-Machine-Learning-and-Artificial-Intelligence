# Improving National Financial Well-Being Through Machine Learning and Artificial Intelligence
<p align="center"><img src="https://therightnewsnetwork.com/wp-content/uploads/2019/03/TRNN-Revelations-Finance.png" /></p>

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
- [References](#references)
- [Code and Resources Used](#code-and-resources-used)

## Executive Summary
This report aims to identify solutions and business recommendations to help the Ministry of Social and Family Development (MSF) better identify people with poor financial well-being which tends to lead to unhappiness. 

Singapore is reputed for its high Gross Domestic Product (GDP) per capita, life expectancy, and integrity. Yet, it has a relatively low happiness level at 31st ranking in the world. To improve happiness in Singapore, our team has chosen to target an aspect that the Singapore Government can likely influence and intervene in. The aspect is financial well-being in which freedom to make life choices is a significant factor.

We adopted a data-driven approach to analyze the personal data that the government collects to aggregate a more holistic judgment of one’s financial well-being through the usage of machine learning and artificial intelligence techniques. Data preprocessing steps such as data cleaning, discretization of the target variable (“At risk” and “Not at risk”), and feature selection were performed followed by data transformation before feeding into the machine learning models. Following that, we developed 7 machine learning models for binary classification: logistic regression, decision tree classifier, random forest classifier, XGBoost classifier, support vector classifier, Bernoulli naïve Bayes classifier, and neural network. A 70:30 train-test split was used to partition the data set for evaluation purposes. Hyperparameter tuning was performed on the training data set to extract the optimal performance out of the models.

The 7 machine learning models were evaluated using various evaluation metrics such as accuracy, precision, recall, and F1-Score. F1-Score was chosen as the final criteria for model selection to account for imbalanced data and the cost of misclassification. All the models managed to beat the baseline F1-Score which was calculated to be at 0.3333. Support vector classifier emerged as the winner providing the highest F1-Score at 0.7194. As such, the support vector classifier was chosen to be recommended to MSF. 

A few business recommendations were proposed to utilize the machine learning model. Firstly, ideas were developed to facilitate the application of the proof of concept to the actual dataset and identify the target group to assist. Secondly, to simplify the workflow for MSF’s transition into interacting with our machine learning model, a graphical user interface was proposed. They can directly export the collected data into the system and the candidates who belong to the “At risk” group along with the relevant information will be generated automatically. Thirdly, a longitudinal study can be conducted to evaluate the efficacy of the intervention by MSF.

The limitations of the data and models were discussed. Limitations of models include the tradeoff between interpretability and predictability and the difficulty in predicting social and dynamic systems. Suggested future work includes survey questions reduction using a randomized approach to reduce survey fatigue, change of target variable to happiness or health scores, and shift of focus from freedom to make life’s choices to social support and generosity, and the other factors which performed poorly when used to determine the Happiness Index in Singapore. 

## Introduction
### Business Problem Statement
Singapore is internationally recognized for its booming economy. We have amongst the world’s highest GDP per capita, healthy life expectancy, and integrity in government and businesses. Furthermore, according to the Programme for International Assessment (PISA), we also deliver one of the world’s best education systems. Yet, Singapore is only ranked 31st in terms of happiness according to the World Happiness Report (WHR) 2020. The top spots were secured by the Nordic countries, with Finland taking 1st place. 

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115279662-9225d600-a179-11eb-9b2e-4647d5337bd4.png" /></p>

In Singapore today, the government has extended various sorts of financial aid to help people suffering from poverty meet their daily needs for food, shelter, and education. These factors form the basis of happiness according to Maslow’s hierarchy of needs as shown above (Dr. Saul McLeod, 2009). Despite these government interventions, it seems that general happiness has not increased.

Out of the six factors that contributed to the Happiness Index score, Singapore ranked 1st and 2nd in GDP per capita, healthy life expectancy, and Perception of integrity in society. However, we are only ranked 14th in freedom to make life choices which definitely can be improved. 

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115279830-c5686500-a179-11eb-97ca-1dd95a5b386b.png" /></p>

The freedom to make life’s choices can be linked to a concept known as financial well-being. According to the Consumer Financial Protection Bureau (CFPB), financial well-being is defined as a state of being wherein a person can fully meet their financial obligations, can feel secure in their financial future, and can make choices that allow them to enjoy life (and thus be happy). 

A recent study by the University of Purdue (Andrew T. Jebb, Louis Tay, Ed Diener, Shigehiro Oishi, 2018) also supports the fact that happiness is impacted by financial well-being, beyond just income level and basic day-to-day living expenses. By improving a person’s financial well-being, they will be able to have greater freedom to make life choices, and hence their happiness. 

Therefore, it becomes critical that we can identify people who do not have this financial freedom and are hence unhappy, yet remain undetected by current measures for financial aid from the Singapore Government.

As analytics consultants to the Ministry of Social and Family Development (MSF)/Community Care Endowment Fund (ComCare), our team aims to study the Financial Well-Being Survey dataset to identify the people who are struggling to obtain financial well-being. This allows us to assist policymakers in easier identification of people who are unhappy and highlight potential reasons behind their struggles.

### Significance of the Business Problem 
Overall happiness is a subjective measure that is affected by a myriad of factors, thus it would be unrealistic to tackle the whole nationwide population of unhappiness. This is why we have decided to use financial well-being as a proxy for happiness as this would mean that our target group of people’s level of happiness can be partially fulfilled by financial aid. 

Studies have shown that poor financial well-being can adversely impact one’s physical, mental, and social health which can lead to other negative impacts such as poorer job performance, lower productivity, and absenteeism, all of which could lead to unhappiness (Human Resources Director). The OCBC Financial Wellness Index (OCBC, 2020), which is a measurement of Singapore’s financial wellness, has dropped from 63 in 2019 to 61 in 2020, especially due to the COVID-19 pandemic. With the current pandemic, more people are likely to have poor financial well-being which may result in decreased happiness levels. 

Given that Singapore’s main resource is its population, the government aims to bridge the income gap to ensure that all Singaporeans are given equal opportunities. The improvement of financial well-being will not only improve these people’s happiness but also their productivity and health. 

Therefore, identifying these people with poor financial well-being will allow the government to more efficiently allocate resources such as money, time and effort, towards those who require financial aid the most to achieve greater happiness. Thus, we need a systematic approach to identify these people through machine learning and artificial intelligence techniques. 

### Expected Business Outcome
Through the machine learning models developed by our team, we hope to improve the financial well-being and happiness of individuals in Singapore. This can be done by highlighting individuals that have low financial well-being so that more attention can be given to them, by better adjusting policies and budgeting to make them happier. Additionally, by identifying the underlying causes of their poor financial well-being, we can help to alleviate their pain points and concerns by recommending them for specialized programs to help them get back on their feet.

## Analytics Approach
Today, government agencies have access to a great deal of personal data (both financial and non-financial data) of their citizens, such as phone and traffic records, health records (including genetic records), spending habits, water and electricity consumption, and even online browsing activities.

In Singapore, citizens receive financial aid from various Ministries (MSF, MOM, MOE) according to their per capita income as well as housing type. Yet, we know that financial well-being is unable to be captured fully through basic financial measures such as income or housing. This grey area requires a more sophisticated solution through a data-driven approach.

Using machine learning and artificial intelligence techniques, we aim to leverage data analytics to sieve through the wide array of personal data that the government collects to aggregate a better and more holistic judgment of a person’s financial well-being and hence extend financial help to those struggling to obtain financial well-being and happiness.

## Data Preparation and Understanding
### Data Set Description
To ensure that our model design would be able to cater to the types of citizen profile data that would be used in real-life implementation, the dataset selected for this project was one that the team deemed to have sufficient nexus with the actual data that is likely to be available. The main criteria for a suitable dataset were that it needed to have a qualifying variable to determine financial well-being, as well as a variety of other variables commonly found in government databases. 

The dataset we have selected is from a survey conducted by the Consumer Financial Protection Bureau (CFPB) in 2015. The dataset consists of 217 columns and 6395 rows. The column “FWBscore” is a discrete variable that indicates a person’s financial well-being on a scale of 0-100. 

<p align="center">
  “Financial Well-Being Survey Data” from Kaggle:
  https://www.kaggle.com/anthonyku1031/nfwbs-puf-2016-data 
</p>

The “FWBscore” scale was determined via an extensive research study involving cognitive interviews, factor testing, and psychometric testing to accurately measure a person’s financial well-being according to four key elements: (i) control over day-to-day finances, (ii) capacity to absorb financial shock, (iii) financial freedom to make choices to enjoy life, and (iv) being on track to meet financial goals. The result was a 10 question scale-based scoring system that accorded each person in the dataset a holistic and comprehensive indicator of their financial well-being. Further elaboration on the survey and methods can be found below.

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115283886-8e488280-a17e-11eb-9a51-c5971b3ba68d.png" /></p>

In addition to asking the 10 questions directly related to computing the FWB Score, indirect questions such as financial knowledge, education level, income and employment, family history, financial habits, demographic information were also asked in the survey. 

This dataset provides us with a multitude of personal data available to a government agency as well as a highly accurate proxy for actual financial well-being “FWBscore”. This would allow our team to develop predictive models and test the accuracy of utilizing personal data to predict financial well-being.

### Exploratory Data Analysis 

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115280293-5ccdb800-a17a-11eb-8a05-f2317d5e1b3d.png" /></p>

The original distribution of FWBscore is roughly normally distributed with values ranging from 14-95, mean of 56, and standard deviation of 14. This is to be expected given the extensive research that has gone into formulating the FWBscore scale as well as the relatively large sample size that has been adjusted for minority group representation. Scores related to social science applications are usually normally distributed as well. 
Since the FWBscore is on a scale of 0-100, the negative values are not valid scores. They are arbitrarily encoded to represent certain statuses. Specifically, “-4” indicates that the respondent’s answer was not written to the database and “-1” indicates that the respondent refused to have the score calculated. Also, when transforming the target variable into a binary variable, later on, we may have to rebalance the dataset such that the predictive power of the model is not hampered. 

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115280374-7111b500-a17a-11eb-9802-916b70dfe39a.png" /></p>

The 10 features (survey questions) used to compute the target variable FWBscore were explored as well using histograms. FWB1_4 and FWB2_2 are concentrated towards the last 3 responses (somewhat/very well/completely). FWB1_3, FWB2_1, FWB2_3 and FWB2_4 are concentrated towards the first 3 responses (never/rarely/sometimes). Other features are roughly normally distributed with not much skew. The overall sentiment is that the respondents are quite optimistic about their finances in terms of security and freedom of choice for the present and future. However, the FWBscore's normal distribution would slightly disagree with the respondents' optimism.

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115280411-7bcc4a00-a17a-11eb-800b-f10c652f580a.png" /></p>

All the feature correlations to FWBscore were calculated and sorted. As seen from above, the top 6 negative correlation and top 4 positive correlation are all the direct questions asked in the Financial Well-Being Survey to compute FWBscore.

### Data Preprocessing
#### Data Cleaning
To ensure that the data set is fit for real-life decision-making purposes for financial intervention, the data itself needs to be of high quality. Without quality data, sub-optimal machine learning models might be trained and provide poor or biased predictions. Data cleaning helps to achieve this objective. This is an iterative process and generally has no end to it. We can only do it on a best efforts basis. 

The original dataset we are using is already quite comprehensive and has already been cleaned by the CFPB once. Hence, our data cleaning process will mainly consist of the following steps: removal of missing values, removal of duplicates, data type conversion, and removal of rows with negative FWBscore. In summary, we removed 5 rows and 66 columns to produce a final dataset with 6389 rows and 151 columns.

#### Discretization of the Target Variable
Rather than assign every respondent a Financial Well-Being score, we are more interested in identifying those with a significantly lower score that requires our help. Hence, we decided to alter the target variable (FWBscore) of our dataset from a discrete integer score of 1-100 to a binary variable which indicates those who are at financial risk and those who are not. 

In Singapore, 10-14% of the population struggles with severe financial issues. Using the bottom 14% poverty line as a benchmark, we add a buffer of 6% to make up the bottom 20th percentile of population data as our “At risk” group. The remaining population will naturally belong to the “Not at risk” group requiring minimal financial intervention.

#### Feature Selection
The data set that we are using consists of a rather high number of columns relative to rows. Keeping all these columns might result in the curse of dimensionality where the machine learning models are not designed to handle such high dimensionality. In addition to weaker performance, the models will also take a longer time to train depending on the algorithms used. Running time that increases exponentially with the number of columns fed to the model will cause a problem for us. Therefore, feature selection serves to remove redundant or non-informative columns from the data set.

- Removal of Unique ID Column
- Removal of Columns Used to Adjust for Demographic and Poverty Differences
- Removal of Direct Survey Questions for FWBscore
- Removal of Secondary Survey Score Columns
- Removal of Survey Items with Incomplete Base
- Removal of Survey Items with Sensitive, Vague, or Region Specific Questions
- Removal of Survey Items with Specific Financial Knowledge Questions

#### Data Transformation
Machine learning models implemented using Scikit-Learn and Keras require all the X features and y labels to be numeric. This means that categorical columns need to be cast into some form of numbers before the machine learning models can be trained. Given that we have established the fact that the X features are considered nominal data instead of interval or ordinal, a few nominal encoding techniques are viable.

A popular technique used is one-hot encoding. It is suitable for nominal data where no relationship exists between different levels. For every unique label in each column, one-hot encoding creates a new binary column for that label. Using the column “SWB_1” as an example, it has 9 unique labels: -4, -1, 1, 2, 3, 4, 5, 6, 7. Therefore, 9 new columns (dummy variables) will be created to represent these unique labels. If a row contains “7” as a value for “SWB_1”, the new dummy variable representing “7” will be filled with the value 1 while all the other dummy variables will be filled with the value 0. 

Dummy encoding is similar to one-hot encoding but it aims to avoid the issue of dummy variable trap. It is a scenario where 1 independent variable can be predicted with the help of other independent variables. This leads to multicollinearity where there is a dependency between the independent variables. Using the same column “SWB_1” as an example, one-hot encoding will result in 1 dummy variable with the value 1 and the others 0 for every row. This means that all the dummy variables are perfectly correlated with each other. As long as we know 1 dummy variable is turned on (value = 1), the other related dummy variables for the column are assumed to be turned off (value = 0). Multicollinearity is a problem for machine learning models like linear regression and logistic regression where highly correlated variables result in unreliable and unstable estimates of regression coefficients. Dummy encoding drops one of the dummy variables to avoid the dummy variable trap. The dropped column implicitly acts as the reference group and the coefficients of the remaining dummy variables are interpreted with respect to this reference group.

Binary encoding initially converts the categorical feature into a numeric data type. The numbers are then converted into binary numbers. Each binary value (0/1) in the binary number forms a new binary column. The row with the longest binary number dictates how many new binary columns are needed to encode the feature. This ensures that all the bits of that row can be represented.

The three nominal encoding techniques described above are considered by our team and we decide to move ahead with binary encoding. Firstly, one-hot encoding is unsuitable for logistic regression due to the dummy variable trap which is one of the machine learning models that we intend to use for our classification task. Secondly, one-hot encoding will result in a very sparse matrix as our survey item columns are of high cardinality. The sparse matrix contains a lot of zero values and wastes considerable space. Thirdly, both one-hot encoding and dummy encoding create many binary columns. This can result in high dimensional data where the number of features can exceed the number of observations and lead to the curse of dimensionality. Binary encoding serves as a middle ground where it is more memory efficient than the other encoding schemes.

For the label (“FWBscore”), a simple label encoding scheme was used. The “At risk” group was represented with the value “1” while the “Not at risk” group was represented with the value “0”. No assumption of order is made with regard to the numbers.

#### Data Partitioning
Data partitioning is required for a train-test split evaluation of our machine learning models. This helps to prevent the problem of overfitting. If we train and evaluate our models on the same data, the models would have seen all the data and achieve overly optimistic performance scores. A separate set of data is required to independently evaluate our models on unseen data for a robust estimate of their classification accuracies. The procedure requires splitting our data set into 2 subsets: training dataset and testing dataset. For the split ratio, we use the common 70:30 train-test split. This means that 70% of the data will be used for training the machine learning models while 30% of the data will be used to evaluate the performance of the models. 

## Model Development and Evaluation
### Model Development
#### Choice of Machine Learning Algorithms
The choice of machine learning algorithms to use depends on the business problem and analytics approach. Given that there is a need to predict which candidates belong to the “At risk” or “Not at risk” group and the labels already exist through the CFPB survey, a supervised learning approach would be appropriate. Furthermore, the prediction problem belongs to that of a classification task since we are predicting categories instead of continuous values. Therefore, the machine learning algorithms chosen should be able to support classification tasks.

There are countless machine learning algorithms developed over the years and all of them have their advantages and disadvantages. Generally, there is a tradeoff between interpretability and predictability. Highly interpretable models tend to be linear with well-defined relationships and easy to compute. Models with high predictive power tend to have non-linear relationships and high computation time. Algorithms can also be tree-based, kernel-based, or distance-based. Ensemble methods such as bagging and boosting can be applied to a base model for improved performance as well, usually reducing bias or variance.

To balance between exploring sufficient machine learning algorithms and avoiding excessive computation time, a total of 7 different algorithms are chosen. They are logistic regression, decision tree classifier, random forest classifier, XGBoost classifier, support vector classifier, Bernoulli naive Bayes classifier, and neural network.

#### Hyperparameter Tuning
Hyperparameters have to be pre-defined before training any machine learning model as they cannot be learnt during training time. Hyperparameter tuning seeks to determine the optimal combination of hyperparameters that allows the machine learning model to maximize its performance. The process of choosing the optimal combination of hyperparameters is not a straightforward task. A balance needs to be struck between searching for a wide range of hyperparameters and minimizing search time. There are multiple ways to tune hyperparameters. The two most popular ways are grid search and randomized search.

Grid search creates a grid of possible values for the hyperparameters. Each search iteration attempts a different combination of hyperparameters in a specific order. It fits the model on all the possible combinations of hyperparameters as defined by the user. If there is a huge combination of hyperparameters, computation time will be very long.

Random search also creates a grid of possible values for the hyperparameters. However, each search iteration tries a random combination of hyperparameters from the grid in no specific order. It fits the model for a number of combinations as decided by the user. If 100 iterations are chosen, the searching process is terminated after 100 times even though the entire grid has not been searched yet. The obvious advantage of random search is the faster computation time. A disadvantage is a possibility that the optimal set of hyperparameters might not be discovered. 

A mix of both methods will be utilized for the 7 machine learning algorithms. Grid search will be used for algorithms with only a few hyperparameters such as logistic regression, support vector classifier, and Bernoulli naive Bayes classifier. Random search will be used for the other algorithms with more hyperparameters such as decision tree classifier, random forest classifier, XGBoost classifier, and neural network.

For both grid search and random search, the number of cross-validation trials for each selected combination of hyperparameters will be set at 3. This is to maximize the data that is used for evaluating each combination of hyperparameters and minimize the variance. For the random search, the number of iterations will be set at 100. Refer to Appendix C for the hyperparameters tuned for all 7 algorithms.

### Model Evaluation
#### Confusion Matrix Explanation and its Tradeoffs

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115283286-bd122900-a17d-11eb-9ac0-f8eded78560f.png" /></p>

A confusion matrix can be thought of as a contingency table for evaluating the classification performance of a machine learning model. Essentially, it is a N x N table or matrix where N is the number of classes we are trying to predict. For our data set, there are 2 classes: “At risk” and “Not at risk” encoded by 1 and 0 respectively. Therefore, the value of N is 2 and the table will have 2 rows and 2 columns with a total of 4 values.

For this data set, the true label refers to the correct answer of “FWBscore” in the test data set. It is the ground truth used for evaluating the machine learning models. The predicted label is the prediction output by the machine learning models. The prediction will be either 0 or 1 since it is a binary classification problem. A comparison will be made between the true label and predicted label for each row in the test data set. Various calculations can be made with the comparisons. Specific to the confusion matrix, 4 metrics can be easily calculated. They are true positives, true negatives, false positives, and false negatives. 

True positives are predicted values (“At risk”) correctly predicted as actual positives (“At risk”). True negatives are predicted values (“Not at risk”) correctly predicted as actual negatives (“Not at risk”). False positives are predicted values (“At risk”) incorrectly predicted as actual positives (“At risk”). False negatives are predicted values (“Not at risk”) incorrectly predicted as actual negatives  (“Not at risk”).

True positives and true negatives are correctly classified predictions. If both true positives and true negatives are both 1 on a normalized scale, the machine learning model is perfect and made no prediction errors at all. For this project, it means all the “At risk” candidates are successfully detected and further financial intervention can be planned for them. There are also no “Not at risk” candidates misclassified as “At risk”. This ensures that all the resources provided by MSF are not wasted on candidates that do not require help. False positives and false negatives are incorrectly classified predictions. For this project, false positives and false negatives are both important and have high costs associated with them. As mentioned earlier, false positives generally take up resources that could be provided to those “At risk” candidates who need help. False negatives mean that the workers at MSF would fail to detect the “At risk” candidates and early financial intervention cannot be provided to them. Therefore, a balanced approach is required to reduce both false positives and false negatives to an acceptable level.

#### Choice of Evaluation Metrics
The choice of evaluation metrics is an important decision to make when evaluating the trained machine learning models. Our team will consider 4 key evaluation metrics commonly used by practitioners. They are accuracy, precision, recall, and F1-Score. 

Accuracy is probably the most common evaluation metric used as it is simple to calculate and interpret. It is essentially the proportion of predictions the machine learning model predicted correctly. It is bounded within a range of 0 to 1 with 0 being the worst score and 1 being the best score. Generally, the accuracy score is used when the true positives and true negatives are more important and the class distribution is balanced.

Precision is the proportion of true positives out of those total predicted positives. It is a suitable evaluation metric when the costs of false positives are high.

Recall is the proportion of true positives out of those total actual positives. It is a suitable evaluation metric when the costs of false negatives are high.

F1-Score is the harmonic mean of precision and recall. It is a suitable evaluation metric when the costs of false positives and false negatives are both high. For imbalanced data sets, F1-Score is also better than the accuracy score. It ignores true negatives in its calculation. A data set with a high proportion of one class tends to have an inflated accuracy score as the machine learning model can simply predict the majority class all the time.

Given that the data set is imbalanced with an 80% negative class and both false positives and false negatives are relatively important, our team will use F1-Score as the evaluation metric for choosing the final machine learning model. However, all 4 evaluation metrics will be calculated to examine the tradeoffs between different machine learning models. Refer to the figure below for the formula of these 4 key evaluation metrics. 

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115283964-ab7d5100-a17e-11eb-8abd-e08e94961095.png" /></p>

#### Baseline F1-Score
A baseline F1-Score acts as a lower bound to evaluate the performances of the machine learning models. The baseline model will be defined as the dummy predictor where it will predict the candidate to be “At risk” or 1 all the time. A trained model is usually expected to perform better than the baseline model. Otherwise, there is no reason to use the model.

Given that the baseline model predicts 1 all the time, the precision will be the marginal probability P(y = 1) which equates to 0.2. This is because 20% of the labels belong to the positive class and they will be classified correctly. The other 80% of the labels belong to the negative class and they will naturally be classified wrongly. Specifically, the true positives will be 0.2 and the false positives will be 0.8. Plugging these values into the precision formula will result in the value 0.2.

On the other hand, the recall will be equal to 1. Since the baseline model predicts 1 all the time, it can find out all the “At risk” candidates. It will never predict 0 or “Not at risk”. Therefore, false negatives will not exist and equate to 0 all the time. Specifically, the true positives will be 0.2 and the false negatives will be 0. Plugging these values into the recall formula will result in the value 1.

With the precision and recall values for the baseline model, we can calculate the baseline F1-Score. The calculation is shown below.

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115282980-6573bd80-a17d-11eb-9ddb-805668396913.png" /></p>

Therefore, the benchmark F1-Score is 0.3333 and the model is required to beat this score.

#### Comparison of Evaluation Metrics

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115279086-d49ae300-a178-11eb-9a21-1ff63601f805.png" /></p>

The evaluation metrics table above details the accuracy, precision, recall, and F1-Score for all 7 trained machine learning models. All the scores were evaluated based on the test data set. Logistics regression has the highest accuracy at 0.8821. The neural network has the highest precision at 0.7778. Random forest classifier has the highest recall at 0.8656. The support vector classifier has the highest F1-Score at 0.7194.

As observed, all the machine learning models managed to beat the baseline F1-Score of 0.3333. Given that we will be using the F1-Score as the evaluation metric to discriminate between the best machine learning models, the final model chosen by our team will be the support vector classifier.

## Proposed Business Recommendations
### Recommendation 1: Application of the Proof of Concept to Actual Dataset and Identify Target Group to Assist
In line with our objectives, we believe that our solutions can help policymakers identify the target group of people that are “At risk” of low financial well-being leading to unhappiness. Input variables can be changed to better suit the local context and include any available information that the SingPass database may already contain: MediSave coverage, ethnocultural details, to even BTO status. Once these “At risk” people are identified, the government can offer financial aid tailored to their needs. 

#### Policy Level Implementation
Our findings can impact policy decisions on many fronts and across different ministries. Our team believes that the greatest benefit can be brought about through budget reallocation in a strategic manner towards this target group. We want to be able to improve their financial well-being and happiness without compromising the welfare of others who still require finances for basic needs. This will translate to re-budgeting on areas outside of ComCare.
One of the current measures the Singapore Government has introduced is the SG bonus. This occurs when the Government has a budget surplus for the fiscal year and decides to issue a one-time bonus to all Singaporean adults above 21. The last incident of this was in 2018, in which all Singaporean adults received $100-300 with no income cap. 
However, not everyone may elicit the same response to this measure, especially for those who are better off in terms of financial well-being which may not have much impact on them. Someone who makes $100,000 a year may not appreciate that extra $100 as much as someone who makes $20,000 a year. 
As such, the Singapore Government could optimize the effectiveness of the SG bonus by channeling the funds towards the target “At Risk” group whose happiness can be impacted by more financial measures. This will allow them to give out these bonuses to those who require it more rather than giving it out to everyone when some people do not even require it. 
Subgroup Level Implementation

#### Ministry and Individual Level Implementation
After identifying the “At risk” individuals, the Ministry of Social and Family Development can assign dedicated financial planners to help these people plan their finances properly. While bonuses payout and fine-tuned policies may be able to help the “At risk” individuals in the short term, proper financial planning is required so that these people will be able to eventually achieve the financial freedom required to make their own life choices. Having a group of financial planners specialized in helping these “At risk” individuals will better aid them in the long term as the financial planners are well-versed in the various support schemes that the “At risk” individuals can apply for to achieve financial freedom. These volunteers can be sought from various financial institutions as a form of pro-bono work.

#### Subgroup Level Implementation
People who are identified with a low FWB score are bound to show similar traits. Some of these traits include poor financial knowledge, pessimism, inability to pay utility bills, and a poor outlook on life. As such, focused solutions need to be tailored to bridge the gap between this “At risk” group and the general population. This will raise their FWB score, suggesting greater freedom to make life choices and hence increase the happiness index of Singapore.

In the case of individuals with poor financial knowledge, a recommendation would be to provide classes to educate them on financial terms, investment strategies, saving habits, and various bank schemes. This relevant knowledge would allow them to effectively manage their finances and directly improve their FWB score. 

As both pessimism and a poor outlook on life are aspects of mental health, the government should schedule appointments with psychologists. This provides an avenue to talk about their problems and develop a better mindset, indirectly improving their  FWB score.

Lastly, individuals who are unable to pay their utility bills can be offered alternative options such as alternative billing methods, taking a loan, or extending the deadline for payment, which may potentially improve their ability to manage their finances and hence their FWB score.

### Recommendation 2: Design of Graphical User Interface
Although the solution is functional on its own, we recognize that the users of this solution  (officers of MSF) may have little to no experience with machine learning modeling or programming. Hence, we can both improve user experience as well as enhance efficiency by automating the modeling process and implementing it via a graphical user interface.

The initial population database will be implemented with our aid. However, we have designed a graphical user interface to allow the officers to key in new entries, i.e. new converted citizens, into the system. The system will go through the same data preprocessing and machine learning modeling process on the entire new database as mentioned above and identify individuals that belong to the “At risk” group again. The system will then generate a list of individuals and their respective particulars into a document for the officers. 

Officers can also go back in the system to update the status of financial aid for the “At risk” group, and track the progress and effectiveness of our intervention, much like a doctor’s patient log. This allows for the database to remain relevant even many years later.

<p align="center"><img src="https://user-images.githubusercontent.com/45563371/115284362-2d6d7a00-a17f-11eb-9c1d-088b7054ad43.png" /></p>

### Recommendation 3: Feedback Collection for Accuracy and Longevity
Before extending help to the identified target persons, policymakers can engage them in a general survey to collect data on their general happiness level. After each year of financial aid, evaluation of the usefulness of the solution can be performed by getting these target individuals to gauge their happiness. This will be similar to a longitudinal study where we observe the same individual over different periods to assess the efficacy of our targeted financial intervention.

## Limitations
### Limitations of Data
Firstly, the collected data reflects the mindset of US citizens. Translating the results to reflect the situation of Singaporeans may not be accurate as there could be a vast difference between how the two groups interpret the various questions due to the contrasting culture and background.

Secondly, the utilization of the Likert scale amplifies this ambiguity and leaves plenty of room for broad interpretation.  For example, despite being in the same financial situation, what one person may consider as a luxury good may not be the same for the other person, causing these individuals to score different ratings on the scale.

Thirdly, respondents may have different interpretations of the definition of financial well-being. Hence, this may affect the “correctness” of their scoring scale. Furthermore, since this is a self-administered survey, there may be some limitations on the credibility of the scores that they provide. 

As such, the above factors may affect the accuracy of our models due to the limitations of data in terms of various interpretations and the level of truthfulness provided by respondents. 

### Limitations of Models
#### Tradeoff Between Interpretability and Predictability
The optimal course of action would be to predict which individuals are financially risky and identify the underlying reasons for each individual. When these cases are highlighted to the people working at MSF, they will understand better how to assist these individuals through proper financial planning and lifestyle changes.

However, as with every machine learning model, there is a trade-off between interpretability and predictability. In our case, we place a greater emphasis on models that can accurately predict whether an individual is at financial risk. This means we are not able to identify the specific cause(s) of financial risk for each case as highly accurate models tend to be black box in nature with low interpretability. 

We have chosen this route as we believe predicting correctly is more crucial. These individuals may be in dire circumstances and desperately require aid. Identifying the cause(s) is secondary and can be left to be determined by MSF who possess the domain expertise to ensure that these individuals are well taken care of. Overall, our team has adopted a human-in-the-loop approach where the relevant personnel has to interact with the machine learning models for the optimal action to take.

#### Difficulty in Predicting Social and Dynamic Systems
Predicting human behavior or scores related to human characteristics such as FWBscore is an inherently difficult task. Individuals tend to exhibit random or unpredictable behavior as compared to physical systems such as machinery and equipment. When a social system we hope to predict constantly changes, the validity of prediction remains a question mark. A more challenging aspect of this project is that the policymakers from MSF will interact with the machine learning predictions and act on the respondents accordingly. We are dealing with a dynamic system where the respondents’ reactions to finding out that they belong to the “At risk” group will probably change their behavior. As a result, this might partially invalidate the machine learning model’s future prediction. Unlike weather forecasting where the prediction won’t affect the weather itself, the prediction of dynamic systems such as an individual’ FWBscore will create a feedback loop due to the respondent’s reaction. Public policymakers at MSF will have to grapple with such issues when deploying the machine learning model.

## Future Work
### Experiment with Reduced and Random Feature Selection
Despite performing the feature selection phase, there is still a high number of features relative to the number of samples. This can result in the curse of dimensionality and overfitting issues. To resolve these issues, a subset of the features (e.g. 10%) can be randomly selected to train the model. This serves to reduce the computation time during the training process. 

An experiment can also be designed to test whether any specific features (survey questions) are important in predicting the FWBscore class. The experiment will be run multiple times where each trial uses a different combination of features. If the evaluation metric does not fluctuate excessively, it might indicate the possibility that the survey questions asked need not be extremely specific as long as sufficient questions are asked. Using fewer questions also helps to combat survey fatigue when the respondents become bored, tired, or uninterested in the survey and begin to perform at a substandard level.

### Augmentation of Target Variable
Currently, the four elements of financial well-being are significantly related to monetary capabilities such as the ability to purchase goods and pay down debts. While important, these represent an incomplete or misleading picture regarding the role that money plays in our lives. Money is treated as the main factor in determining financial well-being based on the four elements. A change can be made where money plays more of a supporting role rather than the leading role to explore the other factors that impact happiness. The machine learning problem can be augmented where instead of predicting the FWBscore, other scores such as happiness or health can be predicted. The hypothesis can be tested regarding the utilization of money as a moderating factor, not the main factor, for the happiness or health of a person. Additional data which are less monetary-focused can be collected such as life experience-based questions. This can shed light on whether an individual’s joy and contentment come from leading a fulfilling life with experiences that may or may not be attained with money.

### Shift in Focus to Other Aspects of Happiness Index
Finally, we can shift our focus from freedom to make life’s choices to social support and generosity, and the other factors which performed poorly when used to determine the Happiness Index in Singapore.

## Conclusion
In conclusion, the machine learning model facilitates the data-driven approach to identify people in Singapore who might experience higher levels of unhappiness due to poor financial well-being. We accomplish this by first obtaining a complete and cleaned dataset through various data preprocessing steps. The processed and transformed data was fed into the machine learning models. The support vector classifier performed the best based on F1-Score and was recommended as the model of choice for the Ministry of Social and Family Development (MSF). 

Through the data-driven approach, our solution overcomes human biases and provides accurate and timely information for better decision-making. In the short and medium term, the machine learning model would help MSF to better identify citizens who have poorer financial well-being which leads to unhappiness, and provide them aid. Fine-tuning of policies serves as a long-term measure to improve the happiness index of Singapore. 

## References
Andrew T. Jebb, Louis Tay, Ed Diener, Shigehiro Oishi. (2018). LETTERShttps://doi.org/10.1038/s41562-017-0277-0© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.1Department of Psychological Sciences, Purdue University, West Lafayette, IN, USA. 2Department of Psychology, University of Vi. Retrieved from https://www.nature.com/articles/s41562-017-0277-0.epdf

Dr. Saul McLeod. (2009, December 29). Maslow's Hierarchy of Needs. Retrieved from https://www.simplypsychology.org/maslow.html

Human Resources Director. (n.d.). Why is financial wellness so important? Retrieved from https://www.hcamag.com/asia/specialisation/financial-wellness/why-is-financial-wellness-so-important/178498

OCBC. (2020). The OCBC Financial Wellness Index 2020 is 61, down 2 points from last year. Retrieved from https://www.ocbc.com/simplyspoton/financial-wellness-index.html

Simon Leow. (2020, July 8). Why Is Singapore Not Happier? Retrieved from https://happinessinitiative.sg/why-is-singapore-not-happier/

## Code and Resources Used
**Python:** Version 3.7.10

**Packages:** pandas, numpy, matplotlib, seaborn, category_encoders, tensorflow, xgboost, keras, sklearn

**Dataset:** https://www.kaggle.com/anthonyku1031/nfwbs-puf-2016-data
