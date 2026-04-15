# DEMOGORGN Antarctica
Code for developing and accessing geostatistical Antarctic bed elevation models 

DEMOGORGN (Digital Elevation Models of Geostatistical ORiGiN) is an ensemble of geostatistically simulated bed topographies. This ensemble is hosted on Source Cooperative (https://source.coop/englacial/demogorgn). Here are instructions for uploading data to Source Coop, making updates, and downloading the data.

DEMOGORGN is a living data product, which means we are continuing to make updates and improvements. Please see our blog for information on DEMOGORGN updates: https://www.gatorglaciology.com/demogorgn. 

## Upload data to Source Coop

We uploaded 100 realizatons of geostatistically simulated bed topography for all of Antarctica on the Bedmap3 500-meter grid to source.coop. The data is currently stored as a single netCDF file and the icechunk is still in progress.

## Download data

The data can be programmatically accessed from source.coop and this is demonstrated in the *data_access.ipynb* notebook. This process will be improved once the data is stored as zarr/icechunk on source.coop

<img src="./figures/whole_realizations.png" width="800"/>

<img src="./figures/mean_std_of_20.png" width="800"/>
