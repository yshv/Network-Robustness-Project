# Network-Toolkit

## Dependencies
### NetworkToolkit
To run most of the scripts, the following dependencies need to be installed:
* pymongo
  `conda install -c conda-forge pymongo`
* pandas
  `conda install -c conda-forge pandas`
* networkx
  `conda install -c conda-forge networkx`
* numpy
  `conda install -c conda-forge numpy`
* matplotlib
  `conda install -c conda-forge matplotlib`
* sklearn
  `conda install scikit-learn`
* progress bar - handy little progress bar library
  `conda install -c conda-forge progress`
* dask
`conda install -c conda-forge dask`
* tqdm
`conda install -c conda-forge tqdm`
* ray 
`pip install -U ray`
* cffi
`conda install cffi`
* mip
`pip install mip`

### ML and GDL
* tensorflow-gpu
`conda install tensorflow-gpu`
* gdl-cuda
`conda install -c dglteam dgl-cuda10.1`

## Description
The package code resides in NetworkToolkit, where most the useful functions will be. Main use case with be creating a Network object, which will have a PhysicalLayer, Routing and Demand object inside it's creation. Furthermore, there is a database that lives in barcelona.ee.ucl.ac.uk. Examples of these are below.
## Keeping the Code Up-to-date
To keep the code up to date, one needs to take part in effective git version control management. A normal workflow would consist of pulling the master branch, creating your own branch to work on your features, then to commit your changes as you develop your code (to this branch). When you want to add features to the master branch, merge your branch to the master, then push it to the remote source.
Make sure your local project is up to date. Then develop within your own branch, commit as you make changes. Write tests to test the working of your code, when ready to merge to master, create a merge request and assign **uceeatz** as the assignee.

## Using MongoDB Database
If using the database, you will have to configure the client portname corresponding to the port you are using (depends on which server you are running code). Since the database lives on barcelona.ee.ucl.ac.uk you will need to SSH tunnel if using another server for computation. If running code on barcelona use the following client:
```python
client = pymongo.MongoClient('mongodb://localhost:27017')
```
If using another server configure your ssh tunnel like this:
```
ssh -Nf -L <localportnumber>:localhost:27017 <barcelona>
```
Then configure your client correspondingly:
```python
client = pymongo.MongoClient('mongodb://localhost:<localportnumber>')
```
## Example Database
There are three main methods in Database that are useful:
1. Reading a set of graphs via a query, the results comes as [(graph1, _id1), ..., (graphn, idn)]
```python
import NetworkToolkit as nt
graph_list = read_topology_dataset_list("Topology_Data", "ER", find_dic={"nodes": 8, "mean k": 3})[:100]
```
2. Reading datasets into a pandas dataframe
```python
import NetworkToolkit as nt
df = nt.Database.read_data_into_pandas("Topology_Data", "ER", find_dic={"nodes": 14, "mean k":4})
```
3. Updating data with an id from the graph list
```python 
nt.Database.update_data_with_id("Topology_Data", "real", graph[1], {"$set":{"FF_kSP Capacity Total":capacity, "FF_kSP Capacity Average":capacity_avg}})
```
## Example Network Routing Max
The below code example shows an example of routing uniform bandwidth demands via FF_kSP until a connection is blocked.
```python
import NetworkToolkit as nt
from progress.bar import ShadyBar

    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ER", {})[:1]  # get a list of graphs from database
    bar = ShadyBar("kSP FF Progress", max=len(graph_list))
    for graph in graph_list:  # loop through graphs

        graph_copy = nt.Tools.assign_congestion(graph[0].copy())  # initialising zero congestion
        network = nt.Network.OpticalNetwork(graph_copy, channel_bandwidth=32e9)  # initialising a Network Object
        SNR_list = network.get_SNR_matrix() # getting a list of worst case SNRs based on shortest paths and full occupation
        blocked=False
        i=0
        bandwidth_step = 0.5
        while not blocked:  # until connection is blocked loop
            demand_matrix_bandwidth = network.demand.create_uniform_bandwidth_requests(bandwidth_step*i)  # create the uniform bandwidth demand matrix
            demand_matrix_connection = network.convert_bandwidth_to_connection(demand_matrix_bandwidth, SNR_list.copy())  # convert this bandwidth demand matrix with the worst case SNR list to a matrix of connection requests
            blocked = network.rwa.FF_kSP(demand_matrix_connection)  # using FF-kSP route the demand (answer is saved in network.rwa.wavelengths dictionary)
            i+=1  
            if blocked: # if blocked route the previous config and break
                i -=1
                demand_matrix_bandwidth = network.demand.create_uniform_bandwidth_requests(bandwidth_step*(i-1))
                demand_matrix_connection=network.convert_bandwidth_to_connection(demand_matrix_bandwidth, SNR_list)
                network.rwa.FF_kSP(demand_matrix_connection)
                break
        # calculate capacity (here just an addition of uniform bandwidths)
        capacity = len(list(graph[0].nodes))*(len(list(graph[0].nodes))-1)*bandwidth_step*(i-1)*1e9
        capacity_avg = bandwidth_step*(i-1)*1e9
        #update the database of that graph with the graph id since graph = (graph, _id)
        nt.Database.update_data_with_id("Topology_Data", "real", graph[1], {"$set":{"FF_kSP Capacity Total":capacity, "FF_kSP                   Capacity Average":capacity_avg}})
        bar.next()
    bar.finish()
```
