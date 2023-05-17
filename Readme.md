# ATL
Autonomous Traffic Lights

This project uses SUMO, which can be setup using this [link](https://sumo.dlr.de/docs/Downloads.php)

Running netedit (inside the ATL/traffic directory):  
```
netedit osm.net.xml
```

Running the SUMO simulation with the default traffic modelling:  
```
sumo-gui osm.sumocfg
```

Generating new scenarios:
`automate.sh` can be used to generate new scenarios.
Once you've finalized the network structure that you wish to model traffic on, you are ready to use this script.
To use the script, you will have to set the path variables to all the files (these have been decribed below) and set the parameters for the vehicles. These varibles have been described below
* OSM\_FILE = path to the final network file (.net.xml)
* WEIGHTS\_FILE = path to the three demand modelling files. [Read more about Customized Weights and Random Trips](https://sumo.dlr.de/docs/Tools/Trip.html#customized_weights)
* OUTPUT\_DIR = path to a directory where you wish to store the generated routes
* PRIVATE\_BUS\_WEIGHT\_FILE = path to the three demand modelling file for private busses

Here is the description of the parameters used to create the scenarios:
* CAR\_PERIOD = period of car generation in the network. A period of 1 would imply that a car is generated every second in the simulation
* PED\_PERIOD = period at which pedestrians are generated in the network.
* BUS\_PERIOD = period at which buses are generated in the network.
* PRIVATE\_BUS\_PERIOD = period at which private buses are generated in the network.
* private\_busses\_exist = A boolean value which is used to indicate whether private buses should exist in the simulation or not. 

After these variables are set, you can run `./automate.sh` to generate the route files and start the GUI simulation.


Creating Visualizations:
`generate_visualizations.sh` can be used to create summary files, tripinfo files and generate visualization for a given scenario. Currently it doesn't support generating aggregated visualizations of multiple scenearios, however this can be done by using the visualization programs directly (read more [here](https://sumo.dlr.de/docs/Tools/Visualization.html)). To use `generate_visualizations.sh`, we'll have to set the paths variables to all the required files, which have been described below:

* OSM\_FILE = path to the final network file (.net.xml)
* WEIGHTS\_FILE = path to the three demand modelling files. [Read more about Customized Weights and Random Trips](https://sumo.dlr.de/docs/Tools/Trip.html#customized_weights)
* OUTPUT\_DIR = path to a directory where you wish to store the generated routes
* PRIVATE\_BUS\_WEIGHT\_FILE = path to the three demand modelling file for private busses

The default visualization parameters that have been set for:
* plot\_summary.py: `meanSpeed` is used as default. Other parameters that can be used are given [here](https://sumo.dlr.de/docs/Simulation/Output/Summary.html) 
* plot\_tripinfo\_distributions.py: `duration`. Other parameters that can be used are given [here](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html) 

Plotting graphs:  
```
sumo -c osm.sumocfg --\<output-name> \<xml-file-name>  
python \<path-to-file-in-$SUMO_HOME/tools/visualization> -i \<xml-file-from-above> -m \<measure-plotted> \<other-options>  
```

To look at the list of output options and graph files available, go here:  
https://sumo.dlr.de/docs/sumo.html  
https://sumo.dlr.de/docs/Tools/Visualization.html#plot_trajectoriespy

Generating random trips:  
```
python \<path-to-randomTrips.py> --help         (to see all options available)  
python \<path-to-randomTrips.py> \<options> --weights-prefix \<customized-edge-weights-file> -r \<output-route-files> -o \<output-trip-files>  
```

To look at the options available in randomTrips.py, go here:  
https://sumo.dlr.de/docs/Tools/Trip.html

Run the file pollutions.py to get the correlation results for the any given network

Update the xml file in the main function to get the emission measures , and the correlation results will be stored in pandas dataframe and comparision swill be computed.
