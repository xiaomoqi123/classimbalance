## Abstract
Actionable Warning Identification (AWI) is crucial for improving the usability of static analysis tools. 
Currently, Machine Learning (ML)-based AWI approaches are notably common, which mainly focus on seeking high performance by improving the warning feature extraction and advancing the AWI model training. 
However, these approaches ignore an important fact that the number of actionable warnings is much smaller than that of unactionable warnings in the warning dataset used for the AWI model training (i.e., the class imbalance). 
Learning from such an imbalanced dataset may limit the performance of ML-based AWI approaches. 
To bridge the above gap, we are the first to conduct a comprehensive empirical study to investigate the impact of class imbalance on the ML-based AWI performance, whether class rebalancing methods can improve the ML-based AWI performance, 
and the differences of class rebalancing methods in the ML-based AWI model.
Our empirical study is performed on 9 real-world and large-scale warning datasets, 25 typical class rebalancing methods, and 7 commonly used ML models.
The experimental results show that (1) the class imbalance has a negative impact on the ML-based AWI performance;
(2) 85\% class rebalancing methods can significantly improve the ML-based AWI performance, but 8\% ones do not work in the imbalanced warning datasets;
(3) RandomOverSampler combined with AdaBoost/Random Forest can make the ML-based AWI model achieve optimal performance on \hl{9} warning datasets.
Finally, we provide three practical guidelines that could help refine ML-based AWI approaches.

About item information:
* data.zip: there are nine warning datasets with the associated warning features.
* main.py: it contains the implementation scripts of our empirical study.
* result.zip: it contains the results on original the warning dataset, the class rebalancing learning results, and statistical analysis results.

About reproduction:
* python: 3.6
* scikit-learn: 0.24.1
