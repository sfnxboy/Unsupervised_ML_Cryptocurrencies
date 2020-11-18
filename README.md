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

In [Trial_&Error_finding_centroids](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Demo/Trial_%26Error_finding_centroids.ipynb) we apply the K-means algorithm to a dataset which does not have a predefined number of clusters. The programmer can create models with any number of cluters they would like! As we add more and more clusters to the code, we see some solid clusters break out. However, before we get trigger-happy and increase the clusters further, we should consider when there might be too many clusters. If we have too many, will it even tell us something about the data? If we increase to 100 clusters, that would really fine-tune each group, but with so many clusters, can we even do anything with that?

Recall that unsupervised learning doesn't have a concrete outcome like supervised learning does. We use unsupervised learning to parse data to help us make decisions. So, at what point do we lose the helpfulness of unsupervised learning? With trial and error, this can become unclear and can only get us so far with more complex datasets.

**Elbow Curve**

An easy method for determining the best number for K is the [elbow curve](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Demo/Elbow_curve.ipynb). Elbow curves get their names from their shape: they turn on a specific value, which looks a bit like an elbow! Inertia is one of the most common objective functions to use when creating an elbow curve. While what it's actually doing can get into some pretty complicated math, basically the inertia objective function is measuring the amount of variation in the dataset.  
![image](https://user-images.githubusercontent.com/68082808/99470029-03830380-2912-11eb-8ff3-88b95dbb4849.png)  
Note the shape of the curve on the following graph. At point 0 (top left), the line starts as a steep vertical slope that breaks at point 2, shifts to a slightly horizontal slope, breaks again at point 3, then shifts to a strong horizontal line that reaches to point 10. The angle at point 3 looks like an elbow, which gives this type of curve its name.

### Managing Data Features

When working with a data set that has too many features, one may want to consider dimensionality reduction. Overfitting is a model is a mad idea because if the model is too specific, future datasets that have different trends will be less accurate. Your first idea is to remove a good amount of features so the model won't be run using every column. This is called **feature elimination**. The downside is, once you remove that feature, you can no longer glean information from it. If we want to know the likelihood of people buying school supplies, but we removed the zip code feature, then we'd miss a detail that could help us understand when certain residents tend to purchase school supplies.  
**Feature extraction** combines all features into a new set that is ordered by how well they predict our original variable. In other words, feature extraction reduces the number of dimensions by transforming a large set of variables into a smaller one. This smaller set of variables contains most of the important information from the original large set. Sometimes, you need to use both feature elimination and extraction. For instance, the customer name feature doesn't inform us about whether or not customers will purchase school supplies. So, we would eliminate that feature during the preprocessing stage, then apply extraction on the remaining features.

**Principal Component Analysis**  
PCA is a statistical technique to speed up machine learning algorithms when the number of input features (or dimensions) is too high. PCA reduces the number of dimensions by transforming a large set of variables into a smaller one that contains most of the information in the original large set. PCA is a complicated process to understand, but it is easy to [code](https://github.com/sfnxboy/Unsupervised_ML_Cryptocurrencies/blob/main/Demo/Principal_Component_Analysis.ipynb).

- The first step in PCA is to standardize these features by using the StandardScaler library:  
```iris_scaled = StandardScaler().fit_transform(df_iris)```
- Now that the data has been standardized, we can use PCA to reduce the number of features. The PCA method takes an argument of n_components, which will pass in the value of 2, thus reducing the features from 4 to 2:
```pca = PCA(n_components=2)```
- After creating the PCA model, we apply dimensionality reduction on the scaled dataset:
```iris_pca = pca.fit_transform(iris_scaled)```
- After this dimensionality reduction, we get a smaller set of dimensions called principal components. These new components are transformed into a DataFrame to fit K-means:
```df_iris_pca = pd.DataFrame(data=iris_pca, columns=["principal component 1", "principal component 2"])```

The ```explained_variance_ration``` method defines how much information can be attributed to each component.  
![image](https://user-images.githubusercontent.com/68082808/99477016-a726e080-291f-11eb-9597-4922a9575af6.png)  
This tells us, is that the first principal component contains 72.77% of the variance and the second contains 23.03%. Together, they contain 95.80% of the information. We then create an elbow curve with the generated principal components and find that the K value is three, we initialize a 2-D k-means model, and lastly we plot the clusters.

Lets look at what is happening under the hood. First, center the points by taking the average of the coordinates, and then moving that balance point to zero, this is a simple transformation. Once the points are centered, we create a 2x2 matric that consits of the variance and covariances as so:  
![image](https://user-images.githubusercontent.com/68082808/99476216-21566580-291e-11eb-9a9b-5f4650ebba5d.png)  
Say the matrix comes out to the following:  
![image](https://user-images.githubusercontent.com/68082808/99476256-33380880-291e-11eb-9cdc-8695440958db.png)
This matrix will be used to transform the points from one graph to another by using the numbers to create a formula for our transformation. The top two values of the matrix will correspond to one point and the bottom two values to another. In our example, the formula for the points becomes (6x + 2y, 2x, + 3y).  
We can now plot (x, y) coordinates into this formula to get a new set of points, plotting these points we create a linear transformation.  
![image](https://user-images.githubusercontent.com/68082808/99476380-76927700-291e-11eb-9c8e-14e4b6c47998.png)  
As you can see, the points stretch out in our graph in two directions. One direction moves from southwest to northeast direction while another direction moves from northwest to southeast. These are called eigenvectors. The magnitude that each of eigenvetors stretch is called the eigenvalue. Eigenvalues and eigenvectors are a complicated subject rooted in linear algebra that are beyond the scope of this project. The big takeaway from eigenvectors and eigenvalues is that they show us the spread of the dataset and by how much.  
Now let's put everything together and show what PCA is doing. We'll up the ante a little bit and expand from two to five columns of data. First, take our data that consists of five columns, or features. We put all the data points into a 5x5 covariance matrix. The eigenvectors and eigenvalues are calculated for each of those five columns in the matrix. From the matrix we can produce a list of eigenvectors and corresponding eigenvalues. We pick how many eigenvalues we want to keep and which to drop, taking two will allow us to plot on a 2D plane. The two eigenvalues and eigenvectors will create a plane on which all the points can be plotted. This now narrows down our five features to two and gives us a good snapshot of what the data should look like because we chose the directions the data spread the most. The statistics, linear transformations, and eigenvalues and eigenvectors all illustrate how PCA works. As you saw earlier, it is much easier to code than do all of this math.
