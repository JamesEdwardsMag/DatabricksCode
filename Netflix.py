# Databricks notebook source
!pip install pandas-profiling 

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

netflix = pd.read_csv("/dbfs/FileStore/NetflixData/netflix_titles_2021.csv")
netflix.head()

# COMMAND ----------

#netflix.shape #Data count
#netflix.dtypes #Data types
#netflix.columns #Column 
#netflix.info()
#netflix.describe()
#netflix.describe(include='all')

# COMMAND ----------

netflix.nunique()
miss = netflix.isnull().sum()/len(netflix)*100 #missing data
                        #percentage^
pd.concat([netflix.isnull().sum(), miss], axis = 1, keys = ["Total", "%"])

# COMMAND ----------

netflix_copy = netflix.copy()
netflix_copy = netflix_copy.dropna(subset=['director', 'cast'], how = 'any')
netflix_copy.fillna({"country" : "missing", "rating" : "missing", "duration" : "missing"}, inplace = True)
netflix_copy.isnull().sum()

# COMMAND ----------

from pandas_profiling import ProfileReport

netflix_profile = ProfileReport(netflix)

netflix_profile


# COMMAND ----------

frame = netflix_copy.type.value_counts().to_frame("Value Count")
frame.plot.bar()

# COMMAND ----------

type_show = ["Movie", "TV Show"]
value_count = [frame['Value Count'][0], frame['Value Count'][1]]

plt.pie(value_count, labels=type_show, autopct ="%2.2f%%")
plt.legend(title = "Media types on netflix")

plt.show

# COMMAND ----------

sns.countplot(x=netflix_copy['rating'], orient="v")
plt.xticks(rotation = 90)
plt.show()

# COMMAND ----------

netflix_copy["date_added"] = pd.to_datetime(netflix_copy['date_added'])
netflix_copy["year_added"] = netflix_copy["date_added"].dt.year
netflix_copy["month_added"] = netflix_copy["date_added"].dt.month

netflix_copy.dtypes

# COMMAND ----------

new_genre = netflix_copy['listed_in'].str.split(",",2)

netflix_copy['Genre 1'] = new_genre.str.get(0)
netflix_copy['Genre 2'] = new_genre.str.get(1)
netflix_copy['Genre 3'] = new_genre.str.get(2)

netflix_copy.drop('listed_in', axis =1, inplace = True)

# COMMAND ----------

#Best month to release content?

#Converting month number to month name 
netflix_copy['month_final'] = netflix_copy['month_added'].replace({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'June', 7:'July', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'})
netflix_copy.month_final.value_counts().to_frame('Value_count')

sns.countplot(x=netflix_copy['month_final'], orient='v')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

g1 = netflix_copy['Genre 1'].describe(include=all)
g2 = netflix_copy['Genre 2'].describe(include=all)
g3 = netflix_copy['Genre 3'].describe(include=all)

plt.pie([g1.freq, g2.freq, g3.freq], labels = [g1.top, g2.top, g3.top], autopct="%2.2f%%")
plt.legend(title='Most watched genre on Netflix')
plt.show()

# COMMAND ----------

import seaborn as sns
sns.countplot(x='release_year', data=netflix).set_title('Count plot for Movies with passing Years.')
sns.set(rc={'figure.figsize':(100,20)})
plt.xticks(rotation = 90, fontsize= 30)
plt.yticks(fontsize = 30)
plt.show()

# COMMAND ----------

netflix["director"].value_counts().head(10).plot(kind = "bar")
plt.xticks(rotation = 0, fontsize ="30")
plt.show()

# COMMAND ----------

netflix[(netflix['type'] == 'Movie') & (netflix['rating'] == 'TV-14') & (netflix['country'] == 'Canada')]
