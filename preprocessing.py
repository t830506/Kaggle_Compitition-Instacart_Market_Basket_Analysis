
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[2]:

print("import data")
ordert = pd.read_csv('order_products__train.csv')
orderp = pd.read_csv('order_products__prior.csv')

orders = pd.read_csv('orders.csv')

products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')

# In[4]:

print("merge orders and order_products__prior") 
orders = orders.fillna(value=0)
orders_products = pd.merge(orderp, orders, how='left', on=['order_id'])

del orderp
orders_products.head(10)


# In[11]:

users = pd.read_csv("users.csv")
prd = pd.read_csv("prd.csv")

print("size of prd features :", prd.shape[1])
print("size of users features :", users.shape[1])

f = lambda x: np.array_split(list(x), 1)[0]
g = orders.groupby('user_id')["days_since_prior_order"].apply(f).reset_index() #user_list_days_since_prior_order
users = pd.merge(users, g, how='left', on=['user_id'])


# ### Users and Products

# In[13]:

# groupby users and products in orders_products
# 商品顧客組合變數新增：出現幾次，第一次出現時間，最後一次出現時間，出現在購物籃的順序平均

f = {'order_number':['count', 'min', 'max'],
     'add_to_cart_order':['mean'],
     'order_dow':['mean', 'std'],
     'days_since_prior_order':['mean', 'std'],
     'order_hour_of_day':['mean', 'std']}
usersXprod = orders_products.groupby(["user_id", "product_id"]).aggregate(f).reset_index()
usersXprod.columns = ["user_id", "product_id",
                      "up_orders", "up_first_order", "up_last_order",
                      "up_average_cart_position", 
                      'up_mean_order_dow', 'up_var_order_dow',
                      'up_mean_days_since_prior_order', 'up_var_days_since_prior_order',
                      'up_mean_order_hour_of_day', 'up_var_order_hour_of_day']

usersXprod = usersXprod.fillna(value=0)
usersXprod['user_product'] = usersXprod.product_id + usersXprod.user_id * 100000

print("nb of usersXproducts features :", usersXprod.shape[1])
usersXprod.head()


# ### 合併三個資料(Users, Products, Users and Products)

# In[14]:

print("merge prod, user and usersXprod feature on usersXprod")
usersXprod = pd.merge(usersXprod, prd, how='left', on=["product_id"])
usersXprod = pd.merge(usersXprod, users, how='left', on=["user_id"])

### 顧客購買此商品的比例
### 顧客上次購買之後幾次沒買
### 這個組合出現的次數 / (顧客購買次數-這個組合第一次出現的次數+1)
usersXprod["up_order_rate"] = usersXprod["up_orders"] / usersXprod["user_orders"]
usersXprod["up_orders_since_last_order"] = usersXprod["user_orders"] - usersXprod["up_last_order"]
usersXprod["up_order_rate_since_first_order"] = usersXprod["up_orders"] / (usersXprod["user_orders"] - usersXprod["up_first_order"] + 1)

del prd
del users
usersXprod.head()


# ### 特別的 days_since_prior_order

# In[15]:

prior_order = usersXprod[["up_last_order","up_nb_order","days_since_prior_order","user_product"]]

d= dict()
i = 0
for i, row in enumerate(prior_order.itertuples(), 1):
    i+=1
    if i%2000000 == 0: print ('order row', i)    
    z = row.user_product
    last = int(row.up_last_order)
    now = int(row.up_nb_order)
    prior = sum(row.days_since_prior_order[last: now])
    d[z] = prior

p = pd.DataFrame.from_dict(d, orient = 'index')
del d
del prior_order


# In[17]:

p["user_product"] = p.index
p.columns = ["up_days_since_prior_order", "user_product"]
usersXprod = pd.merge(usersXprod, p, how='left', on=["user_product"])
usersXprod = usersXprod.drop(["up_nb_order","days_since_prior_order","user_product"], axis=1)

del p
usersXprod.head()


# ### 加入reorder in train

# In[18]:

### ordert and orders merge
ordert = pd.merge(ordert, orders, how='left', on=['order_id'])

del orders
ordert.head()


# > reordered=0的商品為原本prior沒見過的，所以在下面結合的時候，也不會有0出現

# In[19]:

ordert_new = ordert[["user_id", "product_id", "reordered"]]
usersXprod = pd.merge(usersXprod, ordert_new, how='left', on=["user_id", "product_id"])
usersXprod.head()


# In[20]:

usersXprod.reordered = usersXprod.reordered.replace(to_replace = "nan", value = 0)
usersXprod.head()

