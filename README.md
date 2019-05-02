### Predicting no-shows for hospital appointments

Prediction of no-shows for hospital appointments based on electronic medical record (EMR) 
using Logistic Regression and fully-connected Neural Networks. 

Implemented using TensorFlow.

Model implemented in [Code/NN_sigmoid.py](https://github.com/mcreduardo/No_Show/blob/master/Code/NN_sigmoid.py). An usage example can be found at the end of the file.


### Dataset

The dataset was provided by a healthcare software company. In order to protect patient privacy, some features were not provided or replaced by dummy variables. This places a limitation to the trained models.

If all data was available (e.g. location and date), other cirscuntancial data (e.g. weather, public transportation availability) could have been used.

It is important to notice that, for a model like this one, it is essential to pay attention on Fairness. Some features might introduce unfair bias to the model. For example, the usage of social economic related data could result in a model that will be unfair towards people of lower income.