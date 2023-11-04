#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df= pd.read_csv("customer_shopping_data.csv")
df.head()


# In[4]:


# To check the count of records grouped by region/branch of the mall
df.groupby("shopping_mall").count()


# In[5]:


# To check the count of records grouped by the product categories

df.groupby("category").count()


# In[6]:


# total sales for each category of product
df.groupby("category").sum()


# In[7]:


#to get the top performing branches

df.sort_values(by = "price", ascending = False)


# In[16]:


top_categories = df['category'].head(10)


# In[8]:


# to get the top selling categories
df = df.sort_values(by='price', ascending=False)


# In[17]:


# Assuming you have a DataFrame 'df' with columns 'shopping_mall', 'category', and 'price'
total_sales_by_category = df.groupby(["shopping_mall", "category"])['price'].sum()


# In[9]:


# to get total sales for each combination of branch and product_category
df.groupby(["shopping_mall", "category"]).sum()


# In[18]:



# Assuming you have the 'total_sales_by_category' Series from step 2
total_sales_by_category.unstack().plot(kind='bar', stacked=True)
plt.xlabel('Shopping Mall')
plt.ylabel('Total Sales')
plt.title('Sales Distribution by Shopping Mall and Category')
plt.legend(title='Category')
plt.show()


# In[20]:


# Assuming you want to create a pie chart for a specific shopping mall (e.g., 'Mall A')
mall_a_sales = total_sales_by_category['Kanyon']
plt.pie(mall_a_sales, labels=mall_a_sales.index, autopct='%1.1f%%')
plt.title('Sales Distribution for Mall A by Category')
plt.show()


# In[ ]:




