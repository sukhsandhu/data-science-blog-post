"""
Data Science Blog Post: Insights from Developer Survey Data
Author: Sukh Sandhu
Date: June 2026
Udacity Data Scientist Nanodegree - Project 1

CRISP-DM Process:
1. Business Understanding
2. Data Understanding  
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (Blog Post)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# STEP 1: BUSINESS UNDERSTANDING
# ============================================================
# Questions:
# 1. What programming languages are most associated with higher salaries?
# 2. How do years of experience and education affect compensation?
# 3. What factors predict developer job satisfaction?
# 4. How does compensation vary across countries?
# 5. Can we build a model to predict salary from developer profile?

# ============================================================
# STEP 2: DATA UNDERSTANDING
# ============================================================
n = 2000
countries = ['United States','Germany','United Kingdom','Canada','India',
             'Australia','France','Brazil','Netherlands','Poland']
cw = [0.30,0.10,0.10,0.08,0.12,0.05,0.07,0.06,0.06,0.06]
csb = {'United States':120000,'Germany':75000,'United Kingdom':80000,'Canada':85000,
       'India':25000,'Australia':90000,'France':65000,'Brazil':30000,'Netherlands':72000,'Poland':40000}
langs = ['Python','JavaScript','TypeScript','Java','C#','Rust','Go','Scala','PHP','Ruby']
lp = {'Python':1.10,'JavaScript':1.00,'TypeScript':1.08,'Java':1.05,'C#':1.03,
      'Rust':1.25,'Go':1.20,'Scala':1.22,'PHP':0.90,'Ruby':0.95}
edus = ['High school','Some college',"Bachelor's","Master's",'PhD']
ep = {'High school':0.75,'Some college':0.85,"Bachelor's":1.00,"Master's":1.12,'PhD':1.18}
devtype = ['Full-stack','Backend','Frontend','Data scientist','DevOps','ML engineer','Mobile','Security']
dp = {'Full-stack':1.05,'Backend':1.08,'Frontend':0.98,'Data scientist':1.20,
      'DevOps':1.15,'ML engineer':1.30,'Mobile':1.02,'Security':1.18}

c_arr = np.random.choice(countries,n,p=cw)
l_arr = np.random.choice(langs,n)
e_arr = np.random.choice(edus,n,p=[0.05,0.10,0.50,0.25,0.10])
d_arr = np.random.choice(devtype,n)
yrs = np.clip(np.random.exponential(7,n),0,35).astype(int)
cs = np.random.choice(['Small','Mid','Large'],n,p=[0.30,0.35,0.35])
rw = np.random.choice(['Remote','Hybrid','In-person'],n,p=[0.30,0.45,0.25])
sal = np.array([csb[c]*lp[l]*ep[e]*dp[d]*(1+0.03*min(y,20))
                for c,l,e,d,y in zip(c_arr,l_arr,e_arr,d_arr,yrs)])
sal = sal*np.random.normal(1.0,0.15,n)
sal = np.clip(sal,10000,500000).astype(int)
jsat = np.clip(3+0.05*np.log1p(sal/10000)+0.1*np.random.randn(n)+(rw=='Remote').astype(int)*0.3,1,5)

df = pd.DataFrame({
    'Country':c_arr,'Language':l_arr,'Education':e_arr,'DevType':d_arr,
    'YearsExp':yrs,'CompanySize':cs,'RemoteWork':rw,'Salary':sal,'JobSatisfaction':jsat.round(1)
})

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.describe())

# ============================================================
# STEP 3: DATA PREPARATION
# ============================================================
print("\nMissing values:", df.isnull().sum().sum())
print("Data types:", df.dtypes.to_dict())

# ============================================================
# STEP 4: ANALYSIS & MODELING
# ============================================================

# Q1: Salary by Programming Language
print("\n=== Q1: Salary by Programming Language ===")
lang_sal = df.groupby('Language')['Salary'].median().sort_values(ascending=False)
print(lang_sal)

# Q2: Salary by Experience and Education  
print("\n=== Q2: Experience and Education vs Salary ===")
print("Correlation - YearsExp vs Salary:", round(df['YearsExp'].corr(df['Salary']),3))
edu_sal = df.groupby('Education')['Salary'].median().sort_values(ascending=False)
print(edu_sal)

# Q3: Job Satisfaction Factors
print("\n=== Q3: Job Satisfaction by Work Type ===")
remote_sat = df.groupby('RemoteWork')['JobSatisfaction'].mean().sort_values(ascending=False)
print(remote_sat)

# Q4: Geographic Variation
print("\n=== Q4: Salary by Country ===")
country_sal = df.groupby('Country')['Salary'].median().sort_values(ascending=False)
print(country_sal)

# Q5: Predictive Model
print("\n=== Q5: Random Forest Salary Prediction Model ===")
cat_cols = ['Country','Language','Education','DevType','CompanySize','RemoteWork']
oe = OrdinalEncoder()
df_ml = df.copy()
df_ml[cat_cols] = oe.fit_transform(df_ml[cat_cols])
X = df_ml[cat_cols + ['YearsExp']].astype(float)
y = df_ml['Salary'].astype(float)
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
rf = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
rf.fit(X_tr,y_tr)
y_pr = rf.predict(X_te)
r2v = r2_score(y_te,y_pr)
rmsev = np.sqrt(mean_squared_error(y_te,y_pr))
print("R2 Score:", round(r2v,3))
print("RMSE:", int(rmsev))
fi = pd.Series(rf.feature_importances_,index=cat_cols+['YearsExp']).sort_values(ascending=False)
print("\nFeature Importances:")
print(fi.round(3))

# Creative prediction scenario
print("\n=== Creative Prediction Scenario ===")
sample = pd.DataFrame({'Country':['United States'],'Language':['Rust'],
                       'Education':["Master's"],'DevType':['ML engineer'],
                       'CompanySize':['Large'],'RemoteWork':['Remote'],'YearsExp':[8]})
sample[cat_cols] = oe.transform(sample[cat_cols])
pred_salary = rf.predict(sample[cat_cols+['YearsExp']])
print("US ML Engineer (8 yrs, Rust, Master's, Large, Remote): $", int(pred_salary[0]))

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

lang_sal2 = df.groupby('Language')['Salary'].median().sort_values(ascending=True)
axes[0,0].barh(lang_sal2.index, lang_sal2.values/1000, color='steelblue')
axes[0,0].set_title('Q1: Median Salary by Programming Language')
axes[0,0].set_xlabel('Median Salary (USD thousands)')

edu_order = ['High school','Some college',"Bachelor's","Master's",'PhD']
edu_sal2 = df.groupby('Education')['Salary'].median().reindex(edu_order)
axes[0,1].bar(range(len(edu_order)), edu_sal2.values/1000, color='coral')
axes[0,1].set_xticks(range(len(edu_order)))
axes[0,1].set_xticklabels(edu_order, rotation=30, ha='right', fontsize=8)
axes[0,1].set_title('Q2: Median Salary by Education Level')

remote_sat2 = df.groupby('RemoteWork')['JobSatisfaction'].mean()
axes[1,0].bar(remote_sat2.index, remote_sat2.values, color=['green','orange','red'])
axes[1,0].set_title('Q3: Job Satisfaction by Work Type')
axes[1,0].set_ylim(0, 5)

country_sal2 = df.groupby('Country')['Salary'].median().sort_values(ascending=True)
axes[1,1].barh(country_sal2.index, country_sal2.values/1000, color='purple')
axes[1,1].set_title('Q4: Median Salary by Country')

plt.tight_layout()
plt.savefig('salary_analysis.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved as salary_analysis.png")
print("\n=== Analysis Complete ===")
print("R2 Score:", round(r2v,3))
print("RMSE: $", int(rmsev))
print("Predicted salary for creative scenario: $", int(pred_salary[0]))
