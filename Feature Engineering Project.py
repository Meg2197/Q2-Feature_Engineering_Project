import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
train = pd.read_csv('Train_Titanic Dataset - Feature Engg Project.csv')
test = pd.read_csv('test.csv')
# Contatenate dataset
df = pd.concat([train, test], ignore_index = True, sort = False)
print(df.head())

nan_counts = df.isna().sum().sort_values(ascending = False)
summary = pd.concat([df.info(), nan_counts], axis=0)
print(summary)

# Mapping Sex bolean (later to use for tree-based models)
df['Is_male'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Check the entries with NaN vales.
df.loc[df['Embarked'].isna(),:]

# Fill the NaN and map Embarked to numerical codes
df.loc[df['Embarked'].isna(), 'Embarked'] = 'S'  # (Encyclopedia titanica)
df['Embarked_code'] = df['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3} ).astype(int)


#Split the “Ticket” into the variables “Ticket_prefix” and “Ticket_number”
# split the column 'Ticket'
Ticket_split = []
Ticket_len = []
for i in df['Ticket'].index:
    splitted = df.loc[i,'Ticket'].split(' ')
    Ticket_split.append(splitted)
    length = len(splitted)
    Ticket_len.append(length)
    
df['Ticket_split'] = Ticket_split
df['Ticket_len'] = Ticket_len

# df[['Ticket_len']].value_counts()
# df.loc[df['Ticket_len'] > 2, :] We can see tome typo errors.

# Create columns Ticket_preffix and Ticket_number
ticket_preffix = []
ticket_number = []
for i in df['Ticket'].index:
    ticket_parts = df.loc[i, 'Ticket_split']
    number = ticket_parts[-1]
    ticket_number.append(number)
    length = df.loc[i, 'Ticket_len'].item()
    if length > 1:
        preffix = ''.join(ticket_parts[0:-1])
        ticket_preffix.append(preffix)
    else:
        ticket_preffix.append('blanck')
        
df['Ticket_preffix'] = ticket_preffix
df['Ticket_number'] = ticket_number
# drop unnecessary columns created
df.drop(columns=['Ticket_split', 'Ticket_len'], inplace=True)

df.Ticket_preffix.unique()

# We can visualize the variety of ticket prefix values and try to find a relationship with the port of embarkation.

df_grouped = df.groupby(['Ticket_preffix', 'Embarked']).agg({'Survived': ['sum', lambda x: x.count() - x.sum()]})
df_grouped.columns = ['Survived', 'Non-Survived']
df_grouped['Count'] = df.groupby(['Ticket_preffix', 'Embarked']).size()
print(df_grouped)

# Here we engineer the new variable “Companions” describing the number of companions or fellow passengers for each individual. 
n_alone , n_duplicates = df['Ticket'].duplicated(keep=False).astype(int).value_counts()
print("Number of duplicated ticket values:", n_duplicates)
print("Number of non-duplicated ticket values:", n_alone)

df['Companions'] = df['Ticket'].duplicated(keep=False).astype(int) * df.groupby('Ticket')['Ticket'].transform('count') -1 
df.loc[df['Companions'] == -1, 'Companions'] = 0
df['Companions'].value_counts()

##Determining Family Size - The new variable “Family size” is created with the sum of variables “SibSp” and “Parch” to describe the family structure aboard for each passenger. 
df['FamilySize'] = df['SibSp'] + df['Parch']
print(df[['SibSp','Parch','FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by = 'Survived'))


print(df.columns.tolist())
# Extracting the title from the name variable and creating a new variable “Title” to capture the social status of the passengers.

import re
print("re imported successfully")

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df['Title'] = df['Name'].apply(get_title)
print(df['Title'].value_counts())

df_grouped = df.groupby(['Title', 'Sex']).agg({'Survived': ['sum', lambda x: x.count() - x.sum()]})
df_grouped.columns = ['Survived', 'Non-Survived']
df_grouped['Count'] = df.groupby(['Title', 'Sex']).size()
print(df_grouped)

# Clean the variable Title.
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'noble')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Mrs')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Check the survival rate for the title cathegories obtained
df_grouped = df.groupby(['Title', 'Sex']).agg({'Survived': ['sum', lambda x: x.count() - x.sum()]})
df_grouped.columns = ['Survived', 'Non-Survived']
df_grouped['Count'] = df.groupby(['Title', 'Sex']).size()
# Calculate the survived/non-survived rate
df_grouped['Survival Rate'] = df_grouped['Survived'] / (df_grouped['Survived'] + df_grouped['Non-Survived'])
# Order the DataFrame by the survival rate in descending order
df_grouped = df_grouped.sort_values('Survival Rate', ascending=False)

print(df_grouped)

## Since People on titanic wiith noble titles had a higher survival rate, we can create a new variable “noble” to capture this information.
df['noble'] = df['Title'].apply(lambda x: 1  if x =='noble'  else 0)
df['noble'].value_counts()

# fill with the median Fare for the Pclass = 3 and Sex= male.
Fare_Pclass3_male = df.loc[(df['Pclass'] == 3) & (df['Sex'] == 'male'), 'Fare'].median()
df.loc[df['PassengerId'] == 1044, 'Fare'] = df.loc[(df['Pclass'] == 3) & (df['Sex'] == 'male'), 'Fare'].median()
print("medium Fare for Sex = male and Pclass=3: ", Fare_Pclass3_male)

# Divide Fare in 10 levels
n = 10
df['Fare_level'] = pd.cut(df['Fare'], n, labels=np.arange(1,n+1))
df['Fare_range'] = pd.cut(df['Fare'], n)

# Calculate survival probability for each fare level
fare_survival_grouped = df.groupby('Fare_level', observed=True).agg({'Fare_range': ['first', 'count'], 'Survived': 'mean'}).reset_index()
fare_survival_grouped.columns = ['Fare_level', 'Fare_range', 'Count', 'Survival_proba']
fare_survival_grouped = fare_survival_grouped.reset_index(drop=True)
print(fare_survival_grouped)


# Calculate the counts
variable = 'Fare_level'
counts = df[variable].value_counts().sort_index()
# Calculate the probability of survival
survived = df[df['Survived'] == 1][variable].value_counts().sort_index()
not_survived = df[df['Survived'] == 0][variable].value_counts().sort_index()
survival_prob = survived / (survived + not_survived)
# Create subplots with shared x-axis
fig, ax1 = plt.subplots()
# Plot the survival probability using scatter plot markers and lines on the left y-axis
ax1.plot(survival_prob.index, survival_prob.values, marker='o', linestyle='-', color='red')
ax1.set_ylabel('Survival Probability', color='black')
ax1.tick_params(axis='y', labelcolor='red')
# Create a twin y-axis for the left side
ax2 = ax1.twinx()
#Plot the count bars on the right y-axis
ax2.bar(counts.index, counts.values, alpha=0.4, width=0.2, color='royalblue')
ax2.set_ylabel('Counts', color='black')
ax2.tick_params(axis='y', labelcolor='blue')
# Set the x-axis label and title
ax1.set_xlabel(variable)
plt.title('Count and Survival Probability')
# Adjust the layout of subplots
plt.tight_layout()
plt.savefig('fare_survival.png', dpi=150, bbox_inches='tight')
print("saved fare_survival.png")

### CABIN INFO. CLEANUP
##This is the dirtiest column, most values are missing and some rows contain more than one value per column 
# (more than one cabin for some passengers). Only 22% of the “Cabin” values were observed (295/1309).
# To help solve this, we begin by splitting this column into three columns: [“Cabin_label”, “Cabin_number”, “N_of_Cabins reserved”].

# get the last cabin if exists
df['Last_cabin'] = df['Cabin'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else 'N')
# get the Cabin_label of the last cabin
pattern = r'([A-Za-z])'
df['Cabin_label'] = df['Last_cabin'].apply(lambda x: re.search(pattern, str(x)).group(1) if x != 'N' else 'N')
# Get the number of the last cabin if exists
pattern = r'(\d+)'
df['Cabin_number'] = df['Last_cabin'].apply(lambda x: re.search(pattern, str(x)) if x != 'N' else -1)
# Get the number of distinct cabins reserved by the passenger.
df['Cabin_count'] = df['Cabin'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
# drop not used columns and columns with NaN values.
df.drop(columns = ['Cabin','Last_cabin', 'Cabin_number'], inplace = True)

## MAPPING AGE GROUPS
# We can create a new variable “Age_group” to capture the age group of each passenger
# Here I chose the ranges 0–13 for kids, 13–30 for young, range 30–45 for a mature group, 45–60 for old.

# Mapping Age
df.loc[df['Age'] <= 13, 'Age_group'] = 0 # kids
df.loc[(df['Age'] > 13) & (df['Age'] <= 30), 'Age_group'] = 1 # young
df.loc[(df['Age'] > 30) & (df['Age'] <= 45), 'Age_group'] = 2 # mature1
df.loc[(df['Age'] > 45) & (df['Age'] <= 60), 'Age_group'] = 3 # old
df.loc[(df['Age'] > 60) & (df['Age'] <= 100), 'Age_group'] = 4 # very old

# Calculate survival probability for each fare level
age_survival_grouped = df.groupby('Age_group').agg({'Age': ['min', 'max'], 'Survived': ['mean', 'count']}).reset_index()
age_survival_grouped.columns = ['Age_group', 'Age_min' , 'Age_max', 'Survival_proba', 'Count']
print(age_survival_grouped)

# Calculate the counts
variable = 'Age_group'
counts = df[variable].value_counts()

# Calculate the probability of survival
survived = df[df['Survived'] == 1][variable].value_counts().sort_index()
not_survived = df[df['Survived'] == 0][variable].value_counts().sort_index()
survival_prob = survived / (survived + not_survived)

# Create subplots with shared x-axis
fig, ax1 = plt.subplots()   

# Plot the survival probability using scatter plot markers and lines on the left y-axis
ax1.plot(survival_prob.index, survival_prob.values, marker='o', linestyle='-', color='red')
ax1.set_ylabel('Survival Probability', color='black')
ax1.tick_params(axis='y', labelcolor='red')    

# Create a twin y-axis for the left side
ax2 = ax1.twinx()

#Plot the count bars on the right y-axis
ax2.bar(counts.index, counts.values, alpha=0.4, width=0.2, color='royalblue')
ax2.set_ylabel('Counts', color='black')
ax2.tick_params(axis='y', labelcolor='blue')

# Set the x-axis label and title
ax1.set_xlabel(variable)
plt.title('Count and Survival Probability')

# Adjust the layout of subplots
plt.tight_layout()
plt.savefig('age_group_vs_Survivalprob.png', dpi=150, bbox_inches='tight')
print("saved age_group_vs_Survivalprob.png")

## Up to this point, we did data cleaning and feature engineering. 
# Some missing values were filled with values obtained from external sources and we applied a conditioned median operation to fill a small number of NaN values. 
# This operation changed the dataset as cleaned and feature engineered ready to be ingested in ML models etc.


