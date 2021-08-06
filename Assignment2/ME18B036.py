# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # General Instructions to students:
# 
# 1. There are 5 types of cells in this notebook. The cell type will be indicated within the cell.
#     1. Markdown cells with problem written in it. (DO NOT TOUCH THESE CELLS) (**Cell type: TextRead**)
#     2. Python cells with setup code for further evaluations. (DO NOT TOUCH THESE CELLS) (**Cell type: CodeRead**)
#     3. Python code cells with some template code or empty cell. (FILL CODE IN THESE CELLS BASED ON INSTRUCTIONS IN CURRENT AND PREVIOUS CELLS) (**Cell type: CodeWrite**)
#     4. Markdown cells where a written reasoning or conclusion is expected. (WRITE SENTENCES IN THESE CELLS) (**Cell type: TextWrite**)
#     5. Temporary code cells for convenience and TAs. (YOU MAY DO WHAT YOU WILL WITH THESE CELLS, TAs WILL REPLACE WHATEVER YOU WRITE HERE WITH OFFICIAL EVALUATION CODE) (**Cell type: Convenience**)
#     
# 2. You are not allowed to insert new cells in the submitted notebook.
# 
# 3. You are not allowed to **import** any extra packages.
# 
# 4. The code is to be written in Python 3.6 syntax. Latest versions of other packages maybe assumed.
# 
# 5. In CodeWrite Cells, the only outputs to be given are plots asked in the question. Nothing else to be output/print. 
# 
# 6. If TextWrite cells ask you to give accuracy/error/other numbers you can print them on the code cells, but remove the print statements before submitting.
# 
# 7. Any runtime failures on the submitted notebook as it is will get zero marks.
# 
# 8. All code must be written by yourself. Copying from other students/material on the web is strictly prohibited. Any violations will result in zero marks.
# 
# 9. The dataset is given as .npz file, and will contain data in  numpy array. 
# 
# 10. All plots must be labelled properly, all tables must have rows and columns named properly.
# 
# 11. You are allowed to use the numpy library to calculate eigen values. All other functions for reconstruction, clustering, etc., should be written from scratch.
# 
# 12. Change the name of the file with your roll no.
# 
# 

# %%
# Cell type : CodeRead

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# **Cell type : TextRead**
# 
# # Problem 5: Life in Lower Dimensions
# 
# You are provided with a dataset of 1797 images - each image is 8x8 pixels and provided as a feature vector of length 64. You will try your hands at transforming this dataset to a lower-dimensional space using PCA, and perform K-means clustering the images in this reduced space.
#  
# 
# 
# 

# %%
# Cell type : CodeWrite
# write the function for PCA and K-means clustering here. 
def PCA(images):
    me=np.mean(images,axis=0)
    st=np.std(images,axis=0)
    loc=np.where(st==0)
    st[loc]=0.00001
    dat=(images-me.reshape(1,-1))#/st.reshape(1,-1)
    mat=np.cov(dat.T)
    val,vec=np.linalg.eig(mat)
    return val,vec,dat
    

def KMeans(k,dhat,dat):
    components=[]
    for i in range(dhat):
        pc=dat.dot(vec.T[i])
        components.append(pc)
    components=np.array(components).T
    rand=np.random.randint(0,len(images),size=k)
    cent=np.zeros(shape=(k,2))
    cent=components[rand]
    d=[]
    min_sum=10e7
    for i in range(100):
        d=[]
        new_class=[]
        for row in components:
            distances=[]
            for center in cent:
                distance=np.sum((row-center)**2)**0.5
                distances.append(distance)
            new_class.append(np.argmin(distances))
            c=np.min(distances)
            d.append(c)
        if min_sum<sum(d):
            break
        new_centroids=[]
        for cl in set(new_class):
            se=components[np.where(new_class==cl)]
            curr_mean=np.mean(se,axis=0)
            new_centroids.append(curr_mean)
        cent=new_centroids
        min_sum=sum(d)
    return new_class,components

def KMeansBestK(dhat,maxval,dat):
    inertia=[]
    for k in range(1,maxval):
        components=[]
        for i in range(dhat):
            pc=dat.dot(vec.T[i])
            components.append(pc)
        components=np.array(components).T
        cent=np.zeros(shape=(k,2))
        cent=components[:k]
        d=[]
        min_sum=10e7
        for i in range(100):
            d=[]
            new_class=[]
            for row in components:
                distances=[]
                for center in cent:
                    distance=np.sum((row-center)**2)**0.5
                    distances.append(distance)
                new_class.append(np.argmin(distances))
                c=np.min(distances)
                d.append(c)
            if min_sum<sum(d):
                break
            new_centroids=[]
            for cl in set(new_class):
                se=components[np.where(new_class==cl)]
                curr_mean=np.mean(se,axis=0)
                new_centroids.append(curr_mean)
            cent=new_centroids
            min_sum=sum(d)
        inertia.append(min_sum)
    return inertia


    

# %% [markdown]
# 
# %% [markdown]
# **Cell type : TextRead**
# 
# # Problem 5
# 
# #### 5a) Run PCA algorithm on the given data-set. Plot the cumulative percentage variance explained by the principal components. Report the number of principal components that contribute to 90% of the variance in the dataset.
# 
# 
# 

# %%
# Cell type : CodeWrite
# write the code for loading the data, running the PCA algorithm, and plotting. 
# (Use the functions written previously.)
data=np.load('Data.npz')
images=data['arr_0']
val,vec,scaled=PCA(images)
val=val[val.argsort()][::-1]
explained_variances = []
for i in range(len(val)):
    explained_variances.append(val[i] / np.sum(val))
#Calculating the Cumulated sum
cumsum=np.array(explained_variances).cumsum()
#plotting
plt.rcParams['figure.figsize']=8,8
plt.plot(np.arange(1,65),cumsum,color='green')
plt.scatter(np.arange(1,65),cumsum,color='red',s=2)
plt.xlabel('Principal Component Number')
plt.ylabel('Cumulative Percent Variance Explained')
plt.xticks(np.arange(0,68,4))
plt.yticks(np.arange(0,1.1,0.1))
plt.title('Cumulative Variance vs Principal components')
plt.show()
com=cumsum[np.where(cumsum<=0.9)[0]]
print(f"The number of principal components less than 90 percent is {len(com)} ")

# %% [markdown]
# ####5b)  Perform reconstruction of data using the dimensionality-reduced data considering the number of dimensions [2,4,8,16]. Report the Mean Square Error (MSE) between the original data and reconstructed data, and interpret the optimal dimensions $\hat{d}$ based on the MSE values.
# 
# 

# %%
# Cell type : CodeWrite
# Write the code for data reconstruction, run the algorithm for dimensions.
d=[2,4,8,16]
mse=[]
for dim in d:
    components=[]
    for i in range(dim):
        pc=scaled.dot(vec.T[i])
        components.append(pc)
    components=np.array(components).T
    im=scaled[:,:dim]
    diff=(components-im)**2
    summed=np.sum(diff)
    ms=summed/(dim*len(scaled))
    mse.append(ms)
for i in range(len(d)):
    print(f"The MSE of dimension {d[i]}  is : {mse[i]}")
dhat=d[np.argmin(mse)]
print(f"The Optimal dimension is  {dhat}")

# %% [markdown]
# ####5c) Apply K-means clustering on the reduced dataset from last subpart (b) (i.e., the $R^{64}$ to $R^\hat{d}$ reduced dataset; pick the initial k points as cluster centers during initialization). Report the optimal choice of K you have made from the set [1...15]. Which method did you choose to find the optimum number of clusters? And explain briefy why you chose that method. Also, show the 2D scatter plot (consider only the first two dimensions of optimal $\hat{d}$) of the datapoints based on the cluster predicted by K-means (use different color for each cluster).
# 

# %%
# Cell type : CodeWrite
# Write the code for dimensionality reduction, run k-means algorithm on the reduced data-set and do plotting.

#May take quite some time. Please be patient
wss=KMeansBestK(dhat,16,scaled)
#Plotting TO find best K
plt.plot(np.arange(1,16),wss,color='green')
plt.scatter(np.arange(1,16),wss,color='red')
plt.xticks(np.arange(1,16))
plt.xlabel('Number of clusters')
plt.ylabel('Sum of mean squared error')
plt.title('Finding Best Number of Clusters')
plt.show()

print(f"\n The Best value of K is at the elbow of the above graph which is at k=10")
# Doing KMeans on the data for k=10 for dhat=16
classes,recondata=KMeans(10,dhat,scaled)

#Plotting The 2 features
fig,ax=plt.subplots(1,1)
colors=['red','yellow','blue','green','black','cyan','violet','purple','gray','brown']
for i in range(10):
    c=recondata[(np.where(np.array(classes)==i))]
    x=c[:,0]
    y=c[:,1]
    ax.scatter(x,y,color=colors[i],s=20)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
plt.show()


# KMeans for 2 dimensions
classes,recondata=KMeans(10,2,scaled)

#Plotting The 2 features
fig,ax=plt.subplots(1,1)
colors=['red','yellow','blue','green','black','cyan','violet','purple','gray','brown']
for i in range(10):
    c=recondata[(np.where(np.array(classes)==i))]
    x=c[:,0]
    y=c[:,1]
    ax.scatter(x,y,color=colors[i],s=20)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
plt.title('KMeans for 2 dimensions')
plt.show()

# %% [markdown]
# ####5d) In the next TextWrite cell, Summarise and explain your observations from the above experiments. Is the PCA+K-means clustering consistent with how your brain would cluster the images?
# 
# 
# %% [markdown]
# **Cell type : TextWrite**
# 
# Report your observations, MSE values and the method used to choose the optimal number of clusters.
# 
# We found out the optimal number of cluster as 10 from the plot using the elbow method. Since we used PCA on the dataset, we could reduce the dimension of the dataset from 64 to 2 eventhough we need to include about 30 dimensions to explain 90 percent of the variance. The method used is called the elbow method which plots a graph having clusters on the x axis and sum of within cluster squared errors. The best k is at the point where na elbow occurs.
# 

