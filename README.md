# Motor-Imagery-Classification
Brain computer interfaces (BCI) are systems that decipher an individual's intents by analyzing their brain signals. These systems have shown the potential to be a solution for improving independent mobility in paralyzed individuals. 

![Alt text](BCI_figure.jpg?raw=true "Title")


Under the motor imagery paradigm, an individual imagines movement of a limb. The system subsequently decodes this signal and provides a class prediction. Most previous work has sought to improve feature extraction and selection methods but has relied on linear classifiers for class prediction. In this work, we examine the use of ensemble learning methods to improve classification accuracies in a BCI paradigm based on 2-class motor imagery. Our results demonstrate that majority voting schemes outperform Linear Discriminant Analysis (LDA) but that other ensemble techniques such as boosting and bagging perform worse. In the future, we aim to employ more robust feature extraction methods such as Filter-bank Common Spatial Patterns to improve our system’s generalizability.

![Alt text](Pipeline.jpg?raw=true "Title")


To run the MATLAB scripts, open run_evaluation and run it in MATLAB.
The full report can be found here: TwoClassMI_GH_BR_CSC2415.pdf

