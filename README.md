# Objectives
YW
* scrape a website for relevant information, store that information to a dataframe and save that dataframe as a csv file
* load in a dataframe and do the following
    * calculate the zscores of a given column
    * calculate the zscores of a point from a given column in the dataframe
    * calculate and plot the pmf and cdf of another column

# Part 1 - Webscraping
* use the following url scrape the first page of results
* for each item get the name of the item
* store the names to a dataframe and save that dataframe to csv then display
    * store the dataframe in the `data` folder in the repo
    * name the file `part1.csv` and make sure that when you write it you set `index=False`
* the head of the dataframe

* it should match the following
<img src="solutions/images/part1.png"/>


```python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
```


```python
url = "https://www.petsmart.com/dog/treats/dental-treats/#page_name=flyout&category=dog&cta=dentaltreat"
```


```python
response = requests.request("GET", url)
response.status_code

```




    200




```python
# scrape the data
web_soup = BeautifulSoup(response.content, 'html.parser')
```


```python
r_list = web_soup.find_all('div', attrs={'class':'product-name'})
r = r_list[0].find('h3').text
```


```python
data = []
for item in r_list:
    product = {}
    product['name'] = item.find('h3').text
    data.append(product)
data
```




    [{'name': 'Greenies Regular Dental Dog Treats'},
     {'name': 'Greenies Petite Dental Dog Treats'},
     {'name': 'Greenies Large Dental Dog Treats'},
     {'name': 'Pedigree Dentastix Large Dog Treats'},
     {'name': 'Greenies 6 Month+ Puppy Petite Dental Dog Treats'},
     {'name': 'Greenies 6 Month+ Puppy Dental Dog Treats'},
     {'name': 'Greenies 6 Month+ Puppy Teenie Dental Dog Treats'},
     {'name': 'Greenies Teenie Dental Dog Treats'},
     {'name': 'Authority® Dental & DHA Stick Puppy Treats Parsley Mint - Gluten Free, Grain Free'},
     {'name': 'Pedigree Dentastix Large Dog Sticks'},
     {'name': 'Milk-Bone Brushing Chews Large Dental Dog Treats'},
     {'name': 'Pedigree Dentastix Small/Medium Dog Sticks'},
     {'name': 'Pedigree Dentastix Triple Action Dental Dog Treats - Variety Pack'},
     {'name': 'WHIMZEES Variety Value Box Dental Dog Treat - Natural, Grain Free'},
     {'name': 'Pedigree Dentastix Mini Dog Sticks'},
     {'name': 'Virbac® C.E.T.® VeggieDent® Tartar Control Dog Chews'},
     {'name': 'Milk-Bone Brushing Chews Dental Dog Treat'},
     {'name': 'Authority® Dental & DHA Rings Puppy Treats Parsley Mint - Gluten Free, Grain Free'},
     {'name': 'Pedigree Dentastix Large Dog Sticks'},
     {'name': 'Greenies Teenie Dog Dental Treats - Blueberry'},
     {'name': 'Pedigree Dentastix Triple Action Small Dog Treats - Fresh'},
     {'name': 'Milk-Bone Brushing Chew Mini Dental Dog Treats'},
     {'name': 'Authority Dental & Multivitamin Large Dog Treats Parsley Mint - Gluten Free, Grain Free'},
     {'name': 'Greenies Aging Care Dental Dog Treats, 27 oz'},
     {'name': 'Authority Dental & Multivitamin Small Dog Treats Parsley Mint - Gluten Free, Grain Free'},
     {'name': 'Authority® Dental & Multivitamin Medium Dog Treats Parsley Mint - Gluten Free, Grain Free'},
     {'name': 'Greenies Regular Dog Dental Treats - Blueberry'},
     {'name': 'Merrick® Fresh Kisses™ Double-Brush Large Dental Dog Treat - Mint Breath Strips'},
     {'name': 'Blue Buffalo Dental Bones Large Dog Treats - Natural'},
     {'name': 'Greenies Teenie Dental Dog Treats - Fresh'},
     {'name': 'WHIMZEES Brushzees Extra Small Dental Dog Treat - Natural, Grain Free'},
     {'name': 'Greenies Grain Free Teenie Dental Dog Treats'},
     {'name': 'Pedigree Dentastix Dual Flavor Large Dental Dog Treats'},
     {'name': 'Greenies Petite Dental Dog Treats - Blueberry'},
     {'name': 'Merrick® Fresh Kisses™ Double-Brush Small Dental Dog Treat - Mint Breath Strips'},
     {'name': 'WHIMZEES Brushzees Medium Dental Dog Treat - Natural, Grain Free'}]




```python
# load the data into a dataframe file
doggy_data = pd.DataFrame(data)
```


```python
# save the data as a csv file
doggy_data.to_csv('dog_dental', index = False)
```


```python
# display df.head()
doggy_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Greenies Regular Dental Dog Treats</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Greenies Petite Dental Dog Treats</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Greenies Large Dental Dog Treats</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Pedigree Dentastix Large Dog Treats</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Greenies 6 Month+ Puppy Petite Dental Dog Treats</td>
    </tr>
  </tbody>
</table>
</div>



# Part 2

load in the csv file located in the `data` folder called `part2.csv`

create a function that calculates the zscores of an array

then calculate the zscores for each column in part2.csv and add them as columns

See below for final result

<img src="solutions/images/part2_df_preview.png"/>


```python
df = pd.read_csv('data/part2.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salaries</th>
      <th>NPS Score</th>
      <th>eventOutcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>44112.0</td>
      <td>-7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>46777.0</td>
      <td>-12.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>50013.0</td>
      <td>50.0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>48983.0</td>
      <td>-13.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>50751.0</td>
      <td>-11.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create a function that calculates the zscores of an array
def get_z(x):
    mean = x.mean()
    std = x.std()
    zscore = []
    for item in x:
        zscore.append((item-mean)/std)
    return zscore
```


```python
# calculate the zscore for each column and store them as a new column with the names used above
df['salaries_zscores'] = get_z(df['salaries'])
df['NPS Score_zscores'] = get_z(df['NPS Score'])
df['eventOutcome_zscores'] = get_z(df['eventOutcome'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salaries</th>
      <th>NPS Score</th>
      <th>eventOutcome</th>
      <th>salaries_zscores</th>
      <th>NPS Score_zscores</th>
      <th>eventOutcome_zscores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>44112.0</td>
      <td>-7.0</td>
      <td>1</td>
      <td>-1.460155</td>
      <td>-0.913522</td>
      <td>-1.103166</td>
    </tr>
    <tr>
      <td>1</td>
      <td>46777.0</td>
      <td>-12.0</td>
      <td>2</td>
      <td>-0.793981</td>
      <td>-1.080668</td>
      <td>-0.668095</td>
    </tr>
    <tr>
      <td>2</td>
      <td>50013.0</td>
      <td>50.0</td>
      <td>5</td>
      <td>0.014926</td>
      <td>0.991947</td>
      <td>0.637118</td>
    </tr>
    <tr>
      <td>3</td>
      <td>48983.0</td>
      <td>-13.0</td>
      <td>0</td>
      <td>-0.242545</td>
      <td>-1.114097</td>
      <td>-1.538237</td>
    </tr>
    <tr>
      <td>4</td>
      <td>50751.0</td>
      <td>-11.0</td>
      <td>6</td>
      <td>0.199405</td>
      <td>-1.047239</td>
      <td>1.072189</td>
    </tr>
  </tbody>
</table>
</div>



# Part 3 
plot 'salaries' and 'NPS Score' on a subplot (1 row 2 columns) 
then repeat this for the zscores

see image below for reference
<img src="solutions/images/part2-plots.png"/>


```python
# plot for raw salaries and NPS Score data goes here
fig = plt.figure(figsize = (15, 7))
ax1 = fig.add_subplot(121)
ax1.set_title('salaries')
ax1.set_xlabel('salaries')
ax1.set_ylabel('Frequency')
ax2 = fig.add_subplot(122)
ax2.set_title('NPS Scores')
ax2.set_xlabel('NPS Scores')
ax2.set_ylabel('Frequency')
ax1.hist(df['salaries'], bins = 10)
ax2.hist(df['NPS Score'], bins = 10)
plt.show()
```


![png](assessment_files/assessment_16_0.png)



```python
# plot for zscores for salaries and NPS Score data goes here
fig = plt.figure(figsize = (15, 7))
ax1 = fig.add_subplot(121)
ax1.set_title('salaries zscores')
ax1.set_xlabel('salaries_zscores')
ax1.set_ylabel('Frequency')
ax2 = fig.add_subplot(122)
ax2.set_title('NPS Scores Z Scores')
ax2.set_xlabel('NPS Scores_zscores')
ax2.set_ylabel('Frequency')
ax1.hist(df['salaries_zscores'], bins = 10)
ax2.hist(df['NPS Score_zscores'], bins = 10)
plt.show()
```


![png](assessment_files/assessment_17_0.png)


# Part 4 - PMF
using the column 'eventOutcomes'

create a PMF and plot the PMF as a bar chart

See image below for referenc

<img src="solutions/images/part4_pmf.png"/>


```python
import collections
counts = collections.Counter(df['eventOutcome'])
total = len(df['eventOutcome'])
print(counts)
pmf = []
for key, val in counts.items():
    pmf.append(round(val/total, 3))
pmf
```

    Counter({4: 666, 7: 661, 3: 636, 0: 624, 6: 622, 1: 608, 2: 592, 5: 591})
    




    [0.122, 0.118, 0.118, 0.125, 0.124, 0.133, 0.127, 0.132]




```python
plt.bar(counts.keys(), pmf)
plt.title('Event Outcome PMF')
plt.xlabel('Event Outcome')
plt.ylabel('Probability')
plt.show()
```


![png](assessment_files/assessment_20_0.png)


# Part 5 - CDF
plot the CDF of Event Outcomes as a scatter plot using the information above

See image below for reference 

<img src="solutions/images/part5_cmf.png"/>


```python
cdf = []
counter = 0
for x in pmf:
    if len(cdf) >= 1:
        cdf.append(x + cdf[counter])
        counter += 1
    else:
        cdf.append(x)
    
cdf
```




    [0.122, 0.24, 0.358, 0.483, 0.607, 0.74, 0.867, 0.999]




```python
plt.scatter(counts.keys(), cdf)
plt.title('Cumulative Mass Function for Event Outcome')
plt.xlabel('Event Outcome')
plt.ylabel('P(E<=N)')
plt.show()
```


![png](assessment_files/assessment_23_0.png)


I feel like that should have worked but dictionaries are apparently dicts


```python
x = list(range(8))
y = []
for k in x:
    y.append(counts[k])
print(y)
proby = []
for item in y:
    proby.append(item/total)
proby
```

    [624, 608, 592, 636, 666, 591, 622, 661]
    




    [0.1248, 0.1216, 0.1184, 0.1272, 0.1332, 0.1182, 0.1244, 0.1322]




```python
plt.scatter(x, np.cumsum(proby))
plt.title('Cumulative Mass Function for Event Outcome')
plt.xlabel('Event Outcome')
plt.ylabel('P(E<=N)')
plt.show()
```


![png](assessment_files/assessment_26_0.png)


# Bonus:
* using np.where find salaries with zscores <= -2.0

* calculate the skewness and kurtosis for the NPS Score column


```python
# find salaries with zscores <= 2.0 
np.where(df['salaries_zscores'] <= 2)
```




    (array([   0,    1,    2, ..., 4997, 4998, 4999], dtype=int64),)




```python
# calculate skewness and kurtosis of NPS Score column
from scipy.stats import kurtosis, skew
# print('Skewness of NPS Score =' skew(df['NPS Score']))
```


```python
skew(df['N'])
```

# run the cell below to convert your notebook to a README for assessment


```python
!jupyter nbconvert --to markdown assessment.ipynb && mv assessment.md README.md
```
