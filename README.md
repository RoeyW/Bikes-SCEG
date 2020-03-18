# Bikes-Bgcn
 ------------------------
 tensorflow version: TF 2.0
 
 ### datasets: (bikes demands in each stations)
 - citibike: https://www.citibikenyc.com/system-data
 - capitalbike: https://www.capitalbikeshare.com/system-data
 ### urban feature:
 + NYC weather data: https://www.kaggle.com/selfishgene/historical-hourly-weather-data
 + Washington weather data: https://www.kaggle.com/marklvl/bike-sharing-dataset
 + holiday data (e.g., workday, holiday):https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays
 
 ### Files:
 + datapreocess.py: Load data from files with a sliding-window
 + bs.py: Main function
 + Egcn.py: Framework for time-evolving station embedding(E-GCN) and community-informed staiton embedding(B-GCN)
 + GCN_layer.py: details for  GCN and Evolve-GCN (*Evolvegcn: Evolving graph convolutional networks for dynamicgraphs. In: AAAI’20*)
 + vaeTL.py: 
    + encoder: latent representation for  time-evolving station embedding and community-informed staiton embedding
    + decoder: *output* stations's demands
 + Cluster.py: cluster stations to communities
 + Metrics.py: MAPE and RMSPE for all stations\ settled stations \ new stations
 
