# Using Machine Learning to Predict Survival of Patients with Heart Failure

## Table of contents
   * [Overview](#Overview)
   * [Project Set Up and Installation](#Project-Set-Up-and-Installation)
   * [Dataset](#Dataset)
   * [Automated ML](#Automated-ML)
   * [Hyperparameter Tuning](#Hyperparameter-Tuning)
   * [Model Deployment](#Model-Deployment)
   * [Screen Recording](#Screen-Recording)
   * [Comments and future improvements](#Comments-and-future-improvements)
   * [Dataset Citation](#Dataset-Citation)
   * [References](#References)

***

## Overview

The current project uses machine learning to predict patients’ survival based on their medical data. 

I create two models in the environment of Azure Machine Learning Studio: one using Automated Machine Learning (i.e. AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. I then compare the performance of both models and deploy the best performing model as a service using Azure Container Instances (ACI).

The diagram below is a visualization of the rough overview of the operations that take place in this project:

![Project Workflow](img/Project_workflow.JPG?raw=true "Project Workflow") 


## Project Set Up and Installation

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks:

- `automl.ipynb`: for the AutoML experiment;
- `hyperparameter_tuning.ipynb`: for the HyperDrive experiment.

The following files are also necessary:

- `heart_failure_clinical_records_dataset.csv`: the dataset file. It can also be taken directly from Kaggle; 
- `train.py`: a basic script for manipulating the data used in the HyperDrive experiment;
- `scoring_file_v_1_0_0.py`: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio; &
- `env.yml`: the environment file which is also downloaded from within Azure Machine Learning Studio.


## Dataset

### Overview

Cardiovascular diseases (CVDs) kill approximately 18 million people globally every year, being the number 1 cause of death globally. Heart failure is one of the two ways CVDs exhibit (the other one being myocardial infarctions) and occurs when the heart cannot pump enough blood to meet the needs of the body. People with cardiovascular disease or who are at high cardiovascular risk need early detection and management wherein Machine Learning would be of great help. This is what this project attempts to do: create an ML model that could help predicting patients’ survival based on their medical data.

The dataset used is taken from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) and -as we can read in the original [Research article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)- the data comes from 299 patients with heart failure collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old.

The dataset contains 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| _age_ | Age of patient | Years (40-95) |
| _anaemia_ | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes |
| _creatinine-phosphokinase_ | Level of the CPK enzyme in the blood | mcg/L |
| _diabetes_ | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes |
| _ejection_fraction_ | Percentage of blood leaving the heart at each contraction | Percentage |
| _high_blood_pressure_ | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes |
| _platelets_ | Platelets in the blood | kiloplatelets/mL	|
| _serum_creatinine_ | Level of creatinine in the blood | mg/dL |
| _serum_sodium_ | Level of sodium in the blood | mEq/L |
| _sex_ | Female (F) or Male (M) | Binary (0=F, 1=M) |
| _smoking_ | Whether the patient smokes or not | Binary (0=No, 1=Yes) |
| _time_ | Follow-up period | Days |
| _DEATH_EVENT_ | Whether the patient died during the follow-up period | Binary (0=No, 1=Yes) |


### Task 
The main task that I seek to solve with this project & dataset is to classify patients based on their odds of survival. The prediction is based on the first 12 features included in the above table, while the classification result is reflected in the last column named _Death event (target)_ and it is either `0` (_`no`_) or `1` (_`yes`_).


### Access

First, I made the data publicly accessible in the current GitHub repository via this link:
[https://raw.githubusercontent.com/dimikara/heart-failure-prediction/master/heart_failure_clinical_records_dataset.csv](https://raw.githubusercontent.com/dimikara/heart-failure-prediction/master/heart_failure_clinical_records_dataset.csv)

and then create the dataset: 

![Dataset creation](img/00.JPG?raw=true "heart-failure-prediction dataset creation")

As it is depicted below, the dataset is registered in Azure Machine Learning Studio:

***Registered datasets:*** _Dataset heart-failure-prediction registered_
![Registered datasets](img/04.JPG?raw=true "heart-failure-prediction dataset registered")

I am also accessing the data directly via:

```
data = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
```
## Automated ML

***AutoML settings and configuration:***

![AutoML settings & configuration](img/50.JPG?raw=true "AutoML settings & configuration")

Below you can see an overview of the `automl` settings and configuration I used for the AutoML run:

```
automl_settings = {"n_cross_validations": 2,
                   "primary_metric": 'accuracy',
                   "enable_early_stopping": True,
                   "max_concurrent_iterations": 4,
                   "experiment_timeout_minutes": 20,
                   "verbosity": logging.INFO
                  }
```

```
automl_config = AutoMLConfig(compute_target = compute_target,
                             task = 'classification',
                             training_data = dataset,
                             label_column_name = 'DEATH_EVENT',
                             path = project_folder,
                             featurization = 'auto',
                             debug_log = 'automl_errors.log,
                             enable_onnx_compatible_models = False
                             **automl_settings
                             )
```

`"n_cross_validations": 2`

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

`"primary_metric": 'accuracy'`

I chose accuracy as the primary metric as it is the default metric used for classification tasks.

`"enable_early_stopping": True`

It defines to enable early termination if the score is not improving in the short term. In this experiment, it could also be omitted because the _experiment_timeout_minutes_ is already defined below.

`"max_concurrent_iterations": 4`

It represents the maximum number of iterations that would be executed in parallel.

`"experiment_timeout_minutes": 20`

This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the value of 20 minutes.

`"verbosity": logging.INFO`

The verbosity level for writing to the log file.

`compute_target = compute_target`

The Azure Machine Learning compute target to run the Automated Machine Learning experiment on.

`task = 'classification'`

This defines the experiment type which in this case is classification. Other options are _regression_ and _forecasting_.

`training_data = dataset`

The training data to be used within the experiment. It should contain both training features and a label column - see next parameter.

`label_column_name = 'DEATH_EVENT'` 

The name of the label column i.e. the target column based on which the prediction is done.

`path = project_folder`

The full path to the Azure Machine Learning project folder.

`featurization = 'auto'`

This parameter defines whether featurization step should be done automatically as in this case (_auto_) or not (_off_).

`debug_log = 'automl_errors.log`

The log file to write debug information to.

`enable_onnx_compatible_models = False`

I chose not to enable enforcing the ONNX-compatible models at this stage. However, I will try it in the future. For more info on Open Neural Network Exchange (ONNX), please see [here](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx).


### Results

During the AutoML run, the _Data Guardrails_ are run when automatic featurization is enabled. As we can see in the screenshot below, the dataset passed all three checks:

![Data Guardrails Checks](img/40B.JPG?raw=true "Data Guardrails Checks")

![Data Guardrails Checks](img/40.JPG?raw=true "Data Guardrails Checks")


#### Completion of the AutoML run (RunDetails widget): 

![AutoML completed](img/21.JPG?raw=true "AutoML completed: RunDetails widget")

#### Best model

After the completion, we can see the resulting models:

![Completed run models](img/Completed_run_models.JPG?raw=true "Completed run models")

In the _Models_ tab, the first model (at the top) is the best model.
You can see it below along with some of its characteristics & metrics:

![Best model](img/Best_model.JPG?raw=true "Best model")

![Best model graphs](img/Best_model_graphs.JPG?raw=true "Best model graphs")

![Best model metrics](img/Best_model2_metrics.JPG?raw=true "Best model metrics")


*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.


## Screen Recording

The screen recording can be found [here]() and it shows the project in action. More specifically, the screencast demonstrates:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response


## Comments and future improvements

* The first factor that could improve the model is increasing the training time. This suggestion might be seen as a no-brainer, but it would also increase costs and this is a limitation that can be very difficult to overcome: there must always be a balance between minimum required accuracy and assigned budget.

* Continuing the above point, it would be great to be able to experiment more with the hyperparameters chosen for the HyperDrive model or even try running it with more of the available hyperparameters, with less time contraints.

* Another thing I would try is deploying the best models to the Edge using Azure IoT Edge and enabling logging in the deployed web apps.

* I would certainly try to deploy the HyperDrive model as well, since the deployment procedure is a bit defferent than the one used for the AutoML model.

* In the original Research article where this dataset was used it is mentioned that:
> _Random Forests [...] turned out to be the top performing classifier on the complete dataset_

I would love to further explore on this in order to create a model with higher accuracy that would give better and more reliable results, with potential practical benefits in the field of medicine.  

* The question of how much training data is required for machine learning is always valid and, by all means, the dataset used here is rather small and geographically limited: it contains the medical records of only 299 patients and comes from only a specific geographical area. Increasing the sample size can mean higher level of accuracy and more reliable results. Plus, a dataset including data from patients from around the world would also be more reliable as it would compensate for factors specific to geographical regions.

* Finally, although cheerful and taking into account gender equality, it would be great not to have issues like these:

![Notebook not available](img/09.JPG?raw=true "Notebook not available")

![Notebook not available](img/10.JPG?raw=true "Notebook not available")


## Dataset Citation

Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. [BMC Medical Informatics and Decision Making 20, 16 (2020)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5).


## References

- Udacity Nanodegree material
- Research article: [Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)
- [Heart Failure Prediction Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
- [Deploy machine learning models to Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli)
- [A Review of Azure Automated Machine Learning (AutoML)](https://medium.com/microsoftazure/a-review-of-azure-automated-machine-learning-automl-5d2f98512406)
- [The Holy Bible of Azure Machine Learning Service. A walk-through for the believer (Part 3)](https://santiagof.medium.com/the-holy-bible-of-azure-machine-learning-service-a-walk-through-for-the-believer-part-3-74fb7393fffc)
- [What is Azure Container Instances (ACI)?](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-overview)
- [AutoMLConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
- [Using Azure Machine Learning for Hyperparameter Optimization](https://dev.to/azure/using-azure-machine-learning-for-hyperparameter-optimization-3kgj)
- [hyperdrive Package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)
- [Tune hyperparameters for your model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
- [Configure automated ML experiments in Python](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
- [How Azure Machine Learning works: Architecture and concepts](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture)
- [Configure data splits and cross-validation in automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits)
- [How Much Training Data is Required for Machine Learning?](https://machinelearningmastery.com/much-training-data-required-machine-learning/)
- [How Can Machine Learning be Reliable When the Sample is Adequate for Only One Feature?](https://www.fharrell.com/post/ml-sample-size/)
- [Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-137)
- [Predicting sample size required for classification performance](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8)