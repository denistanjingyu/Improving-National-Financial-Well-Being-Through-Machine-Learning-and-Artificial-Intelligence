# Improving National Financial Well-Being Through Machine Learning and Artificial Intelligence

![image](https://therightnewsnetwork.com/wp-content/uploads/2019/03/TRNN-Revelations-Finance.png)

Table of Contents :bookmark_tabs:
=================
- [Overview](#overview)
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Chi-Squared Test](#chi-squared-test)
- [Code and Resources Used](#code-and-resources-used)

## Overview
A/B testing in its simplest sense is an experiment on two variants to see which performs better based on a given metric. Typically, two consumer groups are exposed to two different versions of the same thing to see if there is a significant difference in metrics like sessions, click-through rate, and/or conversions.

## Introduction
In this project, we will be using the chi-squared test to validate an A/B test performed by a company to test the effectiveness of a new webpage in increasing conversion rates.

## Dataset
The dataset used to perform the A/B test was taken from [Kaggle](https://www.kaggle.com/zhangluyuan/ab-testing).

This dataset contains the result of an A/B test where two groups, the control group and the treatment group, were exposed to an old webpage and a new webpage respectively. The purpose of this test was to determine if the new webpage resulted in a significant increase in conversions compared to the old webpage. Each row represents a unique user and shows whether they’re in the control or treatment group and whether they converted or not.

## Chi-Squared Test
After the data was cleaned and correctly formatted, the Chi-Squared Test can be performed. This can simply be done by importing stats from the SciPy library. This step calculates both the chi-squared statistic and the p-value.

The p-value was calculated to be 23%. Assuming a 5% level of significance, we can deduce that the p-value is greater than the alpha and that we fail to reject the null hypothesis. In other words, there is no significance in conversions between the old and new webpage.

## Code and Resources Used
**Python:** Version 3.7.4

**Packages:** pandas, numpy, matplotlib, scipy

**Dataset:** https://www.kaggle.com/zhangluyuan/ab-testing
