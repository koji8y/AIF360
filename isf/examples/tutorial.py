# %% [markdown]
# # Tutorial
# ## See the effect of Intersectional Fairness (ISF) technology with RejectOptionClassification and AdultDataset
# In this tutorial, we will detect the intersectional bias of the AdultDataset, improve the intersectional fairness with ISF, and demonstrate its effectiveness.  
# While ISF supports several mitigation methods, we now select RejectOptionClassification (ROC) and extend it for intersectional fairness.  
# We will also compare ISF with ROC to explain ISF is suitable for retaining intersectional fairness.

# %% [markdown]
# #### RejectOptionClassification
# Reject option classification is a postprocessing technique that gives favorable outcomes to unpriviliged groups and unfavorable outcomes to priviliged groups in a confidence band around the decision boundary with the highest uncertainty.
# 
# References.  
# F. Kamiran, A. Karim, and X. Zhang, “Decision Theory for Discrimination-Aware Classification,” IEEE International Conference on Data Mining, 2012.

# %%
#%matplotlib inline

# %%
from pylab import rcParams

# %%
from aif360.algorithms.intersectional.intersectional_fairness import IntersectionalFairness
from isf.utils.common import output_subgroup_metrics, convert_labels, create_multi_group_label
from isf.analysis.intersectional_bias import calc_intersectionalbias, plot_intersectionalbias_compare
from isf.analysis.metrics import check_metrics_combination_attribute, check_metrics_single_attribute

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## Set up AdultDataset
# Download AdultDataset that AIF360 utilizes.  
# Note: To download it manually, follow [README.md in AIF360](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data) to install the data set.  

# %%
#!curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# %%
#!curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

# %%
#!curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

# %%
#!mv adult* /usr/local/lib/python3.7/dist-packages/aif360/data/raw/adult

# %%
from aif360.datasets import AdultDataset

# %%
dataset = AdultDataset()
convert_labels(dataset)
ds_train, ds_test = dataset.split([0.7])

# %% [markdown]
# ### Ensure what attributes are protected  
# To verify intersectional bias, you need to specify two attributes in the Dataset as protected ones.  
# AdultDataset has already specified the following two attributes as protected:  

# %%
dataset.protected_attribute_names

# %% [markdown]
# ## Classification  
# You first build a classification model since ROC is of type postprocessing.  
# In concrete, train a Logistic Regression model with data ds_train, then proceed with classification for data ds_test.  

# %%
from sklearn.preprocessing import StandardScaler

scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(ds_train.features)
y_train = ds_train.labels.ravel()
X_test = scale_orig.transform(ds_test.features)
Y_test = ds_test.labels.ravel()

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# %%
ds_test_classified = ds_test.copy()
pos_ind = np.where(lr.classes_ == ds_train.favorable_label)[0][0]

ds_test_classified.scores = lr.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
ds_test_classified.labels = lr.predict(X_test).reshape(-1, 1)

# %%
# Calculate ds_train_classified for ROC
ds_train_classified = ds_train.copy()
pos_ind = np.where(lr.classes_ == ds_train.favorable_label)[0][0]

ds_train_classified.scores = lr.predict_proba(X_train)[:, pos_ind].reshape(-1, 1)
ds_train_classified.labels = lr.predict(X_train).reshape(-1, 1)

# %% [markdown]
# ### Confirm the model performance  
# #### (1) Measure the performance for classification
# Check the performance for classification using the above classification results.  

# %%
df_acc = pd.DataFrame(columns=['Accuracy','Precision','Recall','F1 score'])

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df_acc.loc['LR Model']=( accuracy_score(y_true=Y_test, y_pred=ds_test_classified.labels),
                   precision_score(y_true=Y_test, y_pred=ds_test_classified.labels),
                   recall_score(y_true=Y_test, y_pred=ds_test_classified.labels),
                   f1_score(y_true=Y_test, y_pred=ds_test_classified.labels) )

# %%
df_acc

# %% [markdown]
# * The model achieves enough accuracy since `Accuracy` is 84%.

# %% [markdown]
# #### (2) Measure disparate impact to see intersectional bias caused by the combination of two attributes, `race` and `sex`
# Check intersectional bias with disparate impact (DI).  

# %%
df_lr_di = calc_intersectionalbias(ds_test_classified, "DispareteImpact")
df_lr_di = df_lr_di.rename(columns={"DispareteImpact": "LR Model"})
df_lr_di

# %% [markdown]
# * The model requires bias mitigation because DIs for groups other than race=0.0 and sex=1.0 are out of the range for fairness. 
# 
#    Supplement:  
#    In the recruitment field in the US, there is a law saying it is fair if the DI is 0.8 or more (and equal to or less than 1.25, the reciprocal of 0.8), so we consider 0.8 as a standard threshold of fairness.  

# %% [markdown]
# ## Run ISF
# Mitigate intersectional bias in this LR model's judgment.  
# You can use the ROC algorithm, a post-processing method, for it.Run the mitigation algorithm of ISF specifying "RejectOptionClassification" as a parameter.  

# %%
ID = IntersectionalFairness('RejectOptionClassification', 'DemographicParity', 
                             accuracy_metric='F1', options={'accuracy_metric_name':'F1', 'metric_ub':0.2, 'metric_lb':-0.2})

# %%
# training
ID.fit(ds_train, dataset_predicted=ds_train_classified)

# %%
# predict
ds_predicted = ID.predict(ds_test_classified)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ### (1) Compare the raw LR model and the model mitigated by ISF  
# Measure and visualize DIs for intersectional bias to check the effect of ISF.  

# %%
combattr_metrics_isf = check_metrics_combination_attribute(ds_test, ds_predicted)[['base_rate', 'selection_rate', 'Balanced_Accuracy']]

# %%
import seaborn as sns
rcParams['figure.figsize'] = 6,2
sns.set(font_scale = 0.7)
plot_intersectionalbias_compare(ds_test_classified,
                                ds_predicted,
                                vmax=2, vmin=0, center=1,
                                title={"right": "LR Model", "left": "ISF(ROC is used)"})

# %% [markdown]
# * Compared to the left-hand quadrant for the LR model with no bias mitigation, DI in each subgroup gets closer to 1.0 in the right-hand quadrant for the model mitigated by ISF.  
# * This indicates that the judgment result of the model with ISF is closer to fair than that without ISF.  

# %% [markdown]
# Next, ensure ISF does not degrade the accuracy of the model.  

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df_acc.loc['ISF(ROC is used)']=( accuracy_score(y_true=Y_test, y_pred=ds_predicted.labels),
                   precision_score(y_true=Y_test, y_pred=ds_predicted.labels),
                   recall_score(y_true=Y_test, y_pred=ds_predicted.labels),
                   f1_score(y_true=Y_test, y_pred=ds_predicted.labels) )

# %%
df_acc

# %% [markdown]
# * `Accuracy` before and after bias mitigation are almost the same.  
# * This indicates ISF can mitigate intersectional fairness with only minor accuracy degradation.  

# %% [markdown]
# ### (2) Comparison of ROC and ISF (ROC is used)
# Now compare the effects between ordinal ROC and ISF-leveraged ROC.  
# Run an ordinal ROC under the same condition as the ISF-leveraged ROC.  

# %%
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
        
ROC = RejectOptionClassification(
            privileged_groups=[{'race':1,'sex':1}],
            unprivileged_groups=[{'race':0,'sex':0},{'race':0,'sex':1},{'race':1,'sex':0}],
            low_class_thresh=0.01, 
            high_class_thresh=0.99,
            num_class_thresh=100, 
            num_ROC_margin=50,
            metric_name='Statistical parity difference',
            metric_ub=0.2,
            metric_lb=-0.2,
            accuracy_metric_name='Balanced Accuracy')

# %%
# training
ROC.fit(ds_train, ds_train_classified)
# predict
ds_predicted_roc = ROC.predict(ds_test_classified)

# %% [markdown]
# #### Check intersectional bias for ROC  

# %%
df_roc_di =calc_intersectionalbias(ds_predicted_roc, "DispareteImpact")
df_roc_di = df_roc_di.rename(columns={"DispareteImpact": "ROC"})
df_lr_di["ROC"]=df_roc_di["ROC"]
df_lr_di

# %% [markdown]
# * Since ROC does not support intersectional bias, DI values for groups are out of the fairness range though they tend to improve.

# %% [markdown]
# #### Compare DI values for `ROC` and `ISF (ROC is used)`
# Finally, compare DI values ordinal ROC and `ISF (ROC is used)` achieve.  

# %%
plot_intersectionalbias_compare(ds_predicted_roc,
                                ds_predicted,
                                vmax=2, vmin=0, center=1,
                                title={"right": "RoC", "left": "ISF(used RoC)"})

# %% [markdown]
# * Compared to `ROC` (an ordinal ROC; the left-hand quadrant), `ISF (ROC is used)` (ROC-leveraged ISF; the right-hand quadrant) achieves bias mitigation more.  
# * You have confirmed ISF helps an ordinal ROC mitigate intersectional bias.  

# %% [markdown]
# #### Compare accuracies for `ROC` and `ISF (ROC is used)`

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df_acc.loc['ROC']=( accuracy_score(y_true=Y_test, y_pred=ds_predicted_roc.labels),
                   precision_score(y_true=Y_test, y_pred=ds_predicted_roc.labels),
                   recall_score(y_true=Y_test, y_pred=ds_predicted_roc.labels),
                   f1_score(y_true=Y_test, y_pred=ds_predicted_roc.labels) )

# %%
df_acc

# %% [markdown]
# * Accuracies for `ISF (ROC is used)` and `ROC` are almost the same.  
# * This indicates extending RoC with ISF does not cause significant accuracy degradation.  

# %%



