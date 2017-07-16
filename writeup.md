# RoboND-Perception-Exercises

This exercise shows how the robot "perceives" the environment and the steps taken calibrate, filter and segment

### What is a point cloud?

Point clouds are digital representations of three dimensional objects. In practical implementations, they can also contain additional metadata for each point, as well as a number of useful methods for operating on the point cloud.

Point Cloud Types:
* PointXYZ
* PointXYZI
* PointXYZRGB
* Normal
* PointNormal


## Point Cloud Filtering

This is the orginal image. The steps below shows how the image is filtered by first converting it to point cloud
![alt text](./images/tabletop.png "table top")

### VoxelGrid Downsampling Filter

A voxel grid filter allows you to downsample the data by taking a spatial average of the points in the cloud confined by each voxel.

It takes in a parameter (leaf size) which determine the size of each voxel. Higher voxel is good as it is less data, but less details. Too high leaf size might cause missing information. Smaller size would cause high processing

![alt text](./images/voxel_downsample.png "voxel downsample")

### Passthrough Filter

The Pass Through Filter works much like a cropping tool, which allows you to crop any given 3D point cloud by specifying an axis with cut-off values along that axis. The region you allow to pass through, is often referred to as region of interest.

It takes in 3 parameters for the region of interest: axis, axis minimum, axis maximum

![alt text](./images/passthrough_filter.png "pass through")

### RANSAC
RANSAC is an algorithm, that you can use to identify points in your dataset that belong to a particular model.

It returns the inliers and outliers based on the max_distance set

### ExtractIndices Filter
From the results of RANSAC, the inliers and outliers are extracted and calculated

##### Inliers
![alt text](./images/extracted_inliers.png "inliers")

##### Outliers

![alt text](./images/extracted_outliers.png "outliers")

### Outlier Removal Filter

One of the filtering techniques used to remove such outliers is to perform a statistical analysis in the neighborhood of each point, and remove those points which do not meet a certain criteria. PCL’s StatisticalOutlierRemoval filter is an example of one such filtering technique. For each point in the point cloud, it computes the distance to all of its neighbors, and then calculates a mean distance.

By assuming a Gaussian distribution, all points whose mean distances are outside of an interval defined by the global distances mean+standard deviation are considered to be outliers and removed from the point cloud.


## Point Cloud Segmentation and Clustering
Segmentation is the process of dividing point cloud data into meaningful subsets using some common property. Clustering is the process of finding similarities among individual points so they may be segmented

### Kmeans
Clustering by dividing the input data into a key number of clusters based on one or more features or properties


### Euclidean Clustering

Clustering where points that are closer to each other are clustered together by making use of a 3D grid subdivison of the space, 

![alt text](./images/rviz_euclidean2.png "outliers")

