
# coding: utf-8

# # Employee Attrition

# In[56]:


#import all required libraries

#Data Analysis
import pandas as pd
import numpy as np
#Visulaization libraries
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Viridis5
import seaborn as sns
import matplotlib.pyplot as plt
import pygal
import plotly
from ggplot import *
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#model developemnt libraries
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn import tree
#
from IPython.display import SVG, display
import graphviz
import warnings
warnings.filterwarnings("ignore")


# ## 1.  Data Collection

# - Source: Kaggle
# - Data: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/home
# - License: 
#     *  Database: https://opendatacommons.org/licenses/odbl/1.0/
#     *  Contents: https://opendatacommons.org/licenses/dbcl/1.0/

# #### Rule of thumb
# * Know all the available dataset for the problem
# * Extract data in a format that can be used
# * Need skills releated to database (query and management), handling unstructured data (text,video etc) and distributed processing.

# In[2]:


data = pd.read_csv('emp_attrition.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


#Dropping columns (intution)
columns = ['DailyRate', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate', 'MonthlyRate',
        'Over18', 'RelationshipSatisfaction', 'StandardHours']
data.drop(columns, inplace=True, axis=1)


# ### 1.1 Get description of data

# Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# 
# Reference link: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html

# In[6]:


#This will give description only for numeric fields
data.describe()


# In[7]:


#To get description of all columns
data.describe(include = 'all')


# ## 2. Data Cleaning

# * Understand meaning of every feature
# * Look for any missing values
# * Find a way to either drop or fill the values
# * Scaling and normalization
# * Character encoding
# * Handle inconsistent entry
# * Use tools like pandas, python, R 

# ### 2.1 Handling missing values

# In[8]:


data.isnull().sum()


# Looks like the best dataset!!! No null values :-)
# 
# #### But what if we have null values ???? Let's see what we can do in that case.
# 
# * Find why that data is missing. Human error or missed during extraction
# * Drop missing values. 
# * Some ways for filling missing values: 
#   - Zero 
#   - Mean ( works with normal distribution )
#   - Random values from same distribution ( works well with equal distribution ) 
#   - Value after missing value (make sense if data set has some logical order)

# ### 2.2 Encode categorical features(in string) as most of the tools works with numbers

# In[9]:


categorical_column = ['Attrition', 'BusinessTravel', 'Department',
                      'Gender', 'JobRole', 'MaritalStatus', 'OverTime']


# In[10]:


data_encoded = data.copy(deep=True)
lab_enc = preprocessing.LabelEncoder()
for col in categorical_column:
        data_encoded[col] = lab_enc.fit_transform(data[col])
        le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        print('Feature', col)
        print('mapping', le_name_mapping)


# In[11]:


data_encoded.head()


# ## 3. Data Exploration

# * Find patterns in data through data visualization
#    -  Univariate analysis 
#       * Continous variables : Histograms, boxplots. This gives us understanding about the central tendency and spread
#       * Categorical variable : Bar chart showing frequency in each category 
#    -  Bivariate analysis
#       * Continous & Continous : Scatter plots to know how continous variables interact with each other
#       * Categorical & categorical : Stacked column chart to show how the frequencies are spread between two  
#         categorical variables
#       * Categorical & Continous : Boxplots, Swamplots or even bar charts
# * Detect outliers
# * Feature engineering 

# ### 3.1 Get data distribution between output classes

# In[12]:


data_encoded['Attrition'].value_counts()


# From the above result, we can find that about 82% of people stick to the company while rest of them quit :-(
# **** Data is unblanced ****

# ### 3.2 Finding correlation between variables

# In[13]:


data_correlation = data_encoded.corr()


# In[14]:


plt.rcParams["figure.figsize"] = [15,10]
sns.heatmap(data_correlation,xticklabels=data_correlation.columns,yticklabels=data_correlation.columns)


# #### Analysis of correlation results (sample analysis)
# 
# - Monthly income is highly correlated with Job level.
# - Job level is highly correlated with total working hours.
# - Monthly income is highly correlated with total working hours.
# - Age is also positively correlated with the Total working hours.
# - Marital status and stock oprion level are negatively correlated

# ### 3.3 Understanding relationship between features and finding patterns in data through visualization
# 
# Popular data visualization libraries in python are:
#      1. Matplotlib
#      2. Seaborn
#      3. ggplot
#      4. Bokeh
#      5. pygal
#      6. Plotly
#      7. geoplotlib
#      8. Gleam
#      9. missingno
#      10. Leather

# ### 3.3.1 Age Analysis
# Finding relationship between age and attrition. 

# In[15]:


#Plot to see distribution of age overall
plt.rcParams["figure.figsize"] = [7,7]
plt.hist(data_encoded['Age'], bins=np.arange(0,80,10), alpha=0.8, rwidth=0.9, color='blue')


# #### Finding based on above plot
# This plot tells that there are more employees in the range of 30 to 40. Approximately 45% of employees fall in this range.

# In[33]:


#We are going to bin age (multiples of 10) to see which age group are likely to leave the company.
#Before that, let us take only employee who are likely to quit.
positive_attrition_df = data_encoded.loc[data_encoded['Attrition'] == 1]
negative_attrition_df = data_encoded.loc[data_encoded['Attrition'] == 0]


# In[17]:


plt.hist(positive_attrition_df['Age'], bins=np.arange(0,80,10), alpha=0.8, rwidth=0.9, color='red')


# #### Findings based on above plot
# - Employees whose age is in the range of 30 - 40 are more likely to quit.
# - Employees in the range of 20 to 30 are also equally imposing the threat to employers.

# ### 3.3.2 Business Travel vs Attrition
# There are 3 categories in this:
#     1. No travel (0).
#     2. Travel Frequently (1).
#     3. Travel Rarely (2).
# Attrition: No = 0 and Yes = 1

# In[18]:


ax = sns.countplot(x="BusinessTravel", hue="Attrition", data=data_encoded)
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()+1))


# #### Findings
# From the above plot it can be inferred that travel can not be a compelling factor for attrition. Employee who travel rarely are likely to quit more

# ### 3.3.3 Department Vs Attrition
# There are three categories in department:
#        1. Human Resources: 0
#        2. Research & Development: 1
#        3. Sales: 2
# Attrition: No = 0 and Yes = 1

# In[19]:


ax = sns.countplot(x="Department", hue="Attrition", data=data_encoded)
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()+1))


# #### Inference:
#     1. 56% of employess from research and development department are likely to quit.
#     2. 38% of employees from sales department are likely to quit.

# ### 3.3.4 Distance from home Vs Employee Attrition

# In[20]:


plt.hist(negative_attrition_df['DistanceFromHome'], bins=np.arange(0,80,10), alpha=0.8, rwidth=0.9, color='red')


# In[21]:


plt.hist(positive_attrition_df['DistanceFromHome'], bins=np.arange(0,80,10), alpha=0.8, rwidth=0.9, color='red')


# #### Findings
# People who live closeby (0-10 miles) are likely to quit more based on the data

# ### 3.3.5 Education vs Attrition
# There are five categories: 
#      1. Below College - 1 
#      2. College - 2
#      3. Bachelor - 3
#      4. Master - 4
#      5. Doctor - 5

# In[22]:


ax = sns.countplot(x="Education", hue="Attrition", data=data_encoded)
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()+1))


# Inference:
#     1. 41% of employees having bachelor's degree are likely to quit.
#     2. 24% of employees having master's are the next in line

# ### 3.3.6 Gender vs Attrition

# In[23]:


df_age = data_encoded.copy(deep=True)
df_age.loc[df_age['Age'] <= 20, 'Age'] = 0
df_age.loc[(df_age['Age'] > 20) & (df_age['Age'] <= 30), 'Age'] = 1
df_age.loc[(df_age['Age'] > 30) & (df_age['Age'] <= 40), 'Age'] = 2
df_age.loc[(df_age['Age'] > 40) & (df_age['Age'] <= 50), 'Age'] = 3
df_age.loc[(df_age['Age'] > 50), 'Age'] = 4


# In[24]:


df_age = pd.DataFrame({'count': df_age.groupby(["Gender", "Attrition"]).size()}).reset_index()
df_age['Gender-attrition'] = df_age['Gender'].astype(str) + "-" + df_age['Attrition'].astype(str).map(str)


# In[25]:


df_age


# Here,
# 
# * Gender - 0 and Attrition - 0 ===> Female employees who will stay
# * Gender - 0 and Attrition - 1 ===> Female employees who will leave
# * Gender - 1 and Attrition - 0 ===> Male employees who will stay
# * Gender - 1 and Attrition - 1 ===> Male employees who will leave

# In[26]:


output_notebook() 

# x and y axes
Gender_Attrition = df_age['Gender-attrition'].tolist()
count = df_age['count'].tolist()

print(count)

# Bokeh's mapping of column names and data lists
source = ColumnDataSource(data=dict(Gender_Attrition=Gender_Attrition, count=count, color=Viridis5))

plot_bar = figure(x_range=Gender_Attrition, plot_height=350, title="Counts")

# Render and show the vbar plot
plot_bar.vbar(x='Gender_Attrition', top='count', width=0.9, color='color', source=source)
show(plot_bar)


# #### Findings
# 
# From the above plot, we can infer that male employees are likely to leave organization as they amount to 63% compared to female who have 36 % attrition rate.

# ### 3.3.7 Job Role Vs Attrition

# Categories in job role:
# * Healthcare Representative : 0 
# * Human Resources : 1
# * Laboratory Technician : 2
# * Manager : 3 
# * Manufacturing Director : 4
# * Research Director : 5
# * Research Scientist : 6
# * Sales Executive : 7 
# * Sales Representative : 8

# In[27]:


df_jrole = pd.DataFrame({'count': data_encoded.groupby(["JobRole", "Attrition"]).size()}).reset_index()


# In[28]:


#Considering attrition case
df_jrole_1 = df_jrole.loc[df_jrole['Attrition'] == 1]


# In[29]:


import pygal
chart = pygal.Bar(print_values=True)
chart.x_labels = map(str, range(0,9))
chart.add('Count', df_jrole_1['count'])
#chart.render()
display(SVG(chart.render(disable_xml_declaration=True)))


# #### Findings:
# Top three roles facing attrition
# - 26% of employees who are likely to quit belong to Laboratory Technician group
# - 24% of employees belong to Sales Executive group
# - 19% of employees belong to Research Scientist group

# ### 3.3.8 Marital Status vs Attrition

# Categories:
#     1. 'Divorced': 0
#     2. 'Married' : 1
#     3. 'Single'  : 2

# In[57]:


#analyzing employees who has positive attrition
init_notebook_mode(connected=True)
cf.go_offline()
positive_attrition_df['MaritalStatus'].value_counts().iplot(kind='bar')


# In[58]:


plotly.__version__ 


# #### Inference:
# Nearly 50 % of the employees who are single are likely to quit

# ### 3.3.9 Monthly Income vs Attrition

# Computes and draws kernel density estimate, which is a smoothed version of the histogram. This is a useful alternative to the histogram for continuous data that comes from an underlying smooth distribution.

# In[59]:


ggplot(aes(x='MonthlyIncome', color='factor(Attrition)'), data=data_encoded) + geom_density()


# Inference:
#     Looks like people who are less likely to leave the company are the ones who are less paid.

# ## 4. Model Development

# ### 4.1 Creating train and testing data

# In[ ]:


input_data = data_encoded.drop(['Attrition'], axis=1)


# In[ ]:


input_data.head()


# In[ ]:


target_data = data_encoded[['Attrition']]


# In[ ]:


target_data.head()


# In[ ]:


input_train, input_test, output_train, output_test = train_test_split(input_data, target_data, test_size=0.2)


# In[ ]:


input_train.head()


# # Bias Mitigation

# In[ ]:


from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from IPython.display import Markdown, display
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions            import get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.reweighing import Reweighing


# In[ ]:


b_train, b_test = train_test_split(data_encoded, test_size=0.2)
b_train.head()


# In[ ]:


cols = list(b_train)
last_index = len(cols)
cols.insert(last_index, cols.pop(cols.index('Attrition')))
new_b_train = b_train.ix[:, cols]


# In[ ]:


privileged_groups = [{'Gender': 0}]
unprivileged_groups = [{'Gender': 1}]
favorable_label = 0 
unfavorable_label = 1


# In[ ]:


new_dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=new_b_train,
                                label_names=['Attrition'],
                                protected_attribute_names=['Gender'],
                                unprivileged_protected_attributes=unprivileged_groups)


# In[ ]:


metric_orig_train = BinaryLabelDatasetMetric(new_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# In[ ]:



new_dataset.protected_attribute_names


# In[ ]:


RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
RW.fit(new_dataset)
dataset_transf = RW.transform(new_dataset)


# In[ ]:


metric_orig_train = BinaryLabelDatasetMetric(dataset_transf, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Modified training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# In[ ]:


d_tr, d_vl, d_ts = dataset_transf.split([0.5,0.8], shuffle=True)


# In[ ]:


d_tr.features


# In[ ]:


d_tr.labels


# ### 4.2 Feature selection using recursive feature elimination and cross-validation and using for logistic model

# In[ ]:


rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=2, scoring='accuracy')
rfecv.fit(input_train, output_train)
print("Number of features selected: %d" % rfecv.n_features_)
print('Features chosen are: %s' % list(input_train.columns[rfecv.support_]))
selected_features = list(input_train.columns[rfecv.support_])


plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


new_logistic_tr_inp = input_train[selected_features]
new_logistic_ts_inp = input_test[selected_features]
logistic_cls = LogisticRegression()
logistic_cls.fit(new_logistic_tr_inp, output_train)
predicted = logistic_cls.predict(new_logistic_ts_inp)
print('Accuray of logistic regression model is ', accuracy_score(output_test, predicted))


# ### 4.3 Decision Tree Classifier and finding important features

# In[ ]:


dt = tree.DecisionTreeClassifier()
dt.fit(input_train, output_train)
predicted = dt.predict(input_test)
print('Accuray of the model is ', accuracy_score(output_test, predicted))


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
#
data = StringIO()
export_graphviz(dt, out_file=data, 
                         filled=True, rounded=True,  
                         special_characters=True)  
dt_graph = pydotplus.graph_from_dot_data(data.getvalue())  
Image(dt_graph.create_png())


# ### 4.4 Random Forest Classifier and finding important features

# In[ ]:


rf = RandomForestClassifier()
rf.fit(input_train, output_train)
predicted = rf.predict(input_test)


# In[ ]:


print('Accuray of the model is ', accuracy_score(output_test, predicted))


# In[ ]:


imp_feature_indices = np.argsort(rf.feature_importances_)[::-1]
feature_imp = rf.feature_importances_
feature_df = pd.DataFrame(feature_imp,index = input_train.columns,
                           columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:


feature_df


# In[ ]:


# feature importance graph
plt.title('Feature importance of random forest classifier')
plt.bar(range(len(imp_feature_indices)), feature_imp[imp_feature_indices],  align="center")
plt.step(range(len(imp_feature_indices)), np.cumsum(feature_imp[imp_feature_indices]), where='mid')


# ### 5. Conclusion

# Solved two questions 
# - Getting factors contributing to attrition : feature importance techniques
# - Model to predict attrition : Built logistic, decision tree and random forest for the same.
#     
# This is base model and a lot more can be done....
# - Try tuning existing model
# - Try other classifiers
# - Try ensemble methods

# In[ ]:


pd.read_csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"), header=FALSE) 


# In[ ]:


import PIL
from PIL import Image


# In[ ]:


im = Image.open("/Users/saishruthi.tn@ibm.com/Desktop/CHECK/raw/aligned/raw/Ss.png")
rgb_im = im.convert('RGB')
rgb_im.save('audacious.jpg')

