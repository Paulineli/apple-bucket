Please read the test_alignment.py for guidance to use causalab package to do interchange intervention. Now, write code to do the following tasks.

Use parser.add_argument to select
- task
- gpu (only use one)
- sample size
- layer 
- model
- batch size
- number of subgrpah K
- (feel free to add more if needed)

(1) Generate a dataset of entity binding samples. Note that this dataset does not include counterfactual, only the input part. The sample should be generated randomly with fixed query index, answer index, active groups, but the query group can be random. 

(2) Now we view each sample in (1) as a node in a graph. An edge exists between two nodes i,j if and only if the interchange intervention results for (i,j) (i serves as input and j serves as counterfatucal) and (j,i) (j serves as input and i serves as counterfatucal) are consistent with the underlying causal model. Bulid this graph. Figure out the best way to store this graph. You may store a mapping from the 1:sample size to the sample, and build the graph using these indices. You can also use other way of storing it. Please note that the graph can be dense (> 50% of edges exist). save the graph and the dataset. 

(3) Check if graph and dataset are given, if given, directly jump to this step. Now, run Spectral Clustering algorithm to partition the graph into K subgraphes. In the dataset, label each sample in the dataset by their cluster indices. Report the ratio between the number of edges and maximum number of possible edges for each subgraph (which is the same as iia of that) 