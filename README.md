# Unsupervised Machine Learning: Cryptocurrencies

## Overview
Unlike supervised machine learning, we use unsupervised machine learning when there is no known output. Instead, unsupervised machine learning is used to find patterns, or groups, in date. For instance, a bank could use customer segmentation to offer credit cards according to demographics. They could group prospective card holders by age, geography, or gender. Perhaps they target generations, like millenials and baby boomers, or use income level or marital status. Its important to note that unsupervised machine learning uses input data only. In this project we will process data for an unsupervised model, group data using clustering and the K-means algorithm, and lastly make the models more efficient using principle component analysis.

In this project we will take a look at the world of cryptocurrency. The popularity of bitcoin has caused a price jump in the currency, which has rendered it out of reach for many new investors. Luckily for them, there are many other cryptocurrencies available at more affordable prices. In this project we will run unsupervised machine learning algorithms into trends related to these cryptocurrencies to determine whether or not it may be worth it to make a worthwhile investment in these currencies. 

### Install Tools

Open your terminal (Anaconda preferably) and activate your Python environment  
Install the Scikit-learn library: ```conda install scikit-learn```  
Install the Python Plotly library: ```conda install plotly```  
Install the hvPlot visualization library: ```conda install -c pyviz hvplot```

## Notes

### What is Unsupervised Machine Learning

In [supervised learning](https://github.com/sfnxboy/Supervised_ML_Credit_Risk), first a model is initiated, or a template for the algorithm is created. Then it will analyze the data and attempt to learn patterns, which is also called fitting and training. After the data has been fit and trained, it will then make predictions. In unsupervised machine learning (UML), there are no paired inputs and outcomes and the model uses the whole dataset as input. UML is used in one of the following two ways:  
- Transform the data to create an intuitive representation for analysis or to use in a nother machine learning module
- Cluster or determine patterns in a grouping of data, rather than to predict a classification

There are generally two types of unsupervised learning, transformations and clustering algorithms. We use **transformations** when we need to take raw data and make it easier to understand. Transformations also can help prepare data so that it can be used for other machine learning algorithms. Transformations can reduce the dimensional representation, which simply means we'll be decreasing the number of features used for the model or analysis. After doing so, the data can either be processed for use in other algorithms or narrowed down so it can be viewed in 2D. We use **clustering algorithms** to group similar objects into clusters. For example, if a cable service wants to group those with similar viewing habits, we would use a clustering algorithm.

There are **challenges** that arise when working with unsupervised machine learning algorithms. Firstly, since there is no specified outcome, there is not analytical way to know if the result is correct. When working with supervised machine learning algorithms we could simply calculate the accuracy and precision of a model, that isn't possible with UML. This leads to issues when decising whether the model has provided any helpful information we can use. The only way to determine what an unsupervised algorithm did with the data is to go through it manually or create visualizations. Since there will be a manual aspect, unsupervised learning is great for when you want to explore the data. Sometimes you'll use the information provided to you by the unsupervised algorithm to transition to a more targeted, supervised model.

### Preprocessing Data

As with supervised learning, data should be preprocessed into a correct format with only numerical values, null value determination, and so forth. The only difference is unsupervised learning doesn't have a target variable—it only has input features that will be used to find patterns in the data. It's important to carefully select features that could help to find those patterns or create groups.

- **Data Selection:** Consider what data is available, what data is missing, and what data can be removed. Data selection entails making good choices about which data will be used.
- **Data Processing:** Data processing involves organizing the data by formatting, cleaning, and sampling it. For instance, say the date column in your dataset has dates in two different formates, we would convert all dates to the same format. 
- **Data Transformation:** Data transformation entails transforming datasets into a simpler format for storage and future use, such as a CSV, spreadsheet, or database file. 

When preparing our data we should consider the following questions:  
- What knowledge do we hope to glean from running an unsupervised learning model on this dataset?
- What data is available? What type? What is missing? What can be removed?
- Is the data in a format that can be passed into an unsupervised learning model?
- Can I quickly hand off this data for others to use?

Take a look at the [shopping_preprocess](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Demo/shopping_preprocess.ipynb) file, where I take the [original shopping data](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Resources/shopping_data.csv), apply preprocessing techniques to prepare it for machine learning algorithms, and [export](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Resources/shopping_data_cleaned.csv) the file.

### Clustering Data

Clustering is a type of unsupervised machine learning that groups data points together. Say we have a dataset of flowers, with four features, sepal width, sepal length, petal width, petal length. One may observe that data points (flowers) with similar features seem to be closer together than data points with dissimilar features. We can use this spatial information to group similar data points together. This group of data points is called a cluster.

**K-Means Algorithm**

K-means is an unsupervised machine learning algorithm used to identify and solve clustering issues. **K** represents how many clusters there will be. These clusters are then determined by the **means** of all points that will belong to that cluster, where belonging to a cluster is based on some similarity or distance measure to a centroid. A **centroid** is a data point that is the arithmetic mean position of all the points on a cluster. The following [code](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Demo/Clustering_iris.ipynb) displays how one may use the K-means algorithm.  
```
# Initializing model with K = 3 (since we already know there are three classes of iris plants)
model = KMeans(n_clusters=3, random_state=5)
model

# Fitting model
model.fit(df_iris)

# Get the predictions
predictions = model.predict(df_iris)
print(predictions)

# Add a new class column to the df_iris
df_iris["class"] = model.labels_
df_iris.head()

# Import visualization libraries
import plotly.express as px
import hvplot.pandas

# Plotting the clusters with three features
fig = px.scatter_3d(df_iris, x="petal_width", y="sepal_length", z="petal_length", color="class", symbol="class", size="sepal_width",width=800)
fig.update_layout(legend=dict(x=0,y=1))
fig.show()
```  
This code outputs a 3D Model  
![image](https://user-images.githubusercontent.com/68082808/99406732-3d291f80-28bc-11eb-85ee-a74048e3746e.png)

Describe the differences between supervised and unsupervised learning, including real-world examples of each.
Preprocess data for unsupervised learning.
Cluster data using the K-means algorithm.
Determine the best amount of centroids for K-means using the elbow curve.
Use PCA to limit features and speed up the model.
