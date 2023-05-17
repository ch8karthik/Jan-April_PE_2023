import os
import sys
import optparse
import configparser
import folium
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary 
import traci
import sumolib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from bokeh.palettes import Spectral5
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, WidgetBox
from bokeh.models import Panel,ColumnDataSource
from bokeh.models.widgets import Tabs,Select
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.io import output_file, show
from bokeh.tile_providers import OSM, get_provider
import math

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options




csv_file = "/Users/karthikmac/Downloads/ATL/traffic/plain.edg.csv"

def getConnectedComponent():
    df = pd.read_csv(csv_file,sep=';',engine='python')
    df = df.drop_duplicates(subset=["edge_id"],keep=False)
    adj_list = {}
    x = len(df)
    for i in range(x):
        val = df.iloc[i]
        if( val["edge_from"] in adj_list):
            adj_list[val["edge_from"]].append(val["edge_to"])
    
        else:
            adj_list[val["edge_from"]] = [val["edge_to"]]
        
    G = nx.DiGraph(adj_list)
    
    return G

def getNumberOfConnectedComponents(G):
    g = nx.number_strongly_connected_components(G) 
    return g

G = getConnectedComponent()

def check_edge(a):
    df = pd.read_csv(csv_file,sep=";",engine='python')
    df = df.drop_duplicates(subset=["edge_id"],keep=False)    
    df = df.set_index("edge_id")
    edges = list(df.index)
    if a not in edges:
        return False
    
    val = df.loc[a]
    edge_from = val["edge_from"]
    edge_to = val["edge_to"]
    res = G.has_edge(edge_from,edge_to)
    
    if(not res):
        return False
    
    G_copy = G.copy()
    G_copy.remove_edge(edge_from,edge_to)
    if(getNumberOfConnectedComponents(G) == getNumberOfConnectedComponents(G_copy)):
        res = True
        G.remove_edge(edge_from,edge_to)
    else:
        res = False
    return res



def getEdgeId(node_from,node_to):    #retreives edge id's of the road                           
    df = pd.read_csv(csv_file,sep=";",engine='python')
    df = df.drop_duplicates(subset=["edge_id"],keep=False)

    df = df.set_index("edge_id")
    edges = df.index
    l = []
    for i in edges:
        val = df.loc[i]
        if(node_from  == val['edge_from']and node_to == val['edge_to']):
            l.append(i)
    return l

result = {}
k = 1
G = getConnectedComponent()



def plot_degree_dist(G):      #To pot histogram bar plot of degrees for nodes
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

def degree_histogram_directed(G, in_degree=False, out_degree=False): 
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


#in_degree_freq = degree_histogram_directed(G, in_degree=True)
#out_degree_freq = degree_histogram_directed(G, out_degree=True)
#degrees = range(len(in_degree_freq))
#plt.figure(figsize=(12, 8)) 
#plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree') 
#plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
#plt.xlabel('Degree')
#plt.ylabel('Frequency')
#plt.show()

idg = nx.in_degree_centrality(G)
odg = nx.out_degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweeness_centrality = nx.edge_betweenness_centrality(G , normalized=True)
deg_centrality = nx.degree_centrality(G)

deg_centrality_sorted = dict(sorted(deg_centrality.items(), key=lambda item: item[1],reverse=True))

#betweeness_centrality_sorted = dict(sorted(betweeness_centrality.items(), key=lambda item: item[1],reverse=True))

N_nodes = G.number_of_nodes()

color_list=N_nodes*['lightsteelblue']

count = 0
for key in betweeness_centrality:
    #print(betweeness_centrality[key])
    count += 1
    if count == 6:
        break



# print(nx.in_degree_centrality(G))

# print(nx.out_degree_centrality(G))

# print(nx.closeness_centrality(G))

nodes = nx.in_degree_centrality(G).keys()
degree = nx.degree_centrality(G).values()
in_degree = nx.in_degree_centrality(G).values()
out_degree = nx.out_degree_centrality(G).values()
closeness = nx.closeness_centrality(G).values()
bet_centrality = nx.betweenness_centrality(G, normalized = True, endpoints = False).values()

cc = nx.closeness_centrality(G)
dfc = pd.DataFrame.from_dict({
    'node': list(cc.keys()),
    'closeness_centrality': list(cc.values())
})

dfc = dfc.sort_values('closeness_centrality', ascending=False)

d = nx.degree_centrality(G)
dfd = pd.DataFrame.from_dict({
    'node': list(d.keys()),
    'degree': list(d.values())
})

dfd = dfd.sort_values('degree', ascending=False)
#od = nx.out_degree_centrality(G)
#dfo = pd.DataFrame.from_dict({
#    'node': list(od.keys()),
#    'outdegree': list(od.values())
#})

#dfo = dfo.sort_values('outdegree', ascending=False)
bc = nx.betweenness_centrality(G, k=231 , normalized = True, endpoints = False)
dfb = pd.DataFrame.from_dict({
    'node': list(bc.keys()),
    'betweeness': list(bc.values())
})
dfb = dfb.sort_values('betweeness', ascending=False)
print(dfb)

df = pd.DataFrame({'nodes': nodes, 'degree': degree, "closeness": closeness, 'betweeness': bet_centrality})
print(df)

print("Correlations:\n")
#print("{}".format(df["in degree"].corr(df["out degree"])))

print("correlation between degree and closeness centrality measures:{}\n".format(df["degree"].corr(df["closeness"])))
print("correlation between degree and betweeness centrality measures:{}\n".format(df["degree"].corr(df["betweeness"])))
#print("correlation between out degree and closeness centrality measures:{}\n".format(df["out degree"].corr(df["closeness"])))
#print("correlation between out degree and betweeness centrality measures:{}\n".format(df["out degree"].corr(df["betweeness"])))
print("correlation between betweeness and closeness centrality measures:{}\n".format(df["betweeness"].corr(df["closeness"])))

#col1 = df['in degree']
#col2 = df['betweeness']
#col3 = df['out degree']
#col4 = df['closeness']
#col5 = df['nodes']

#plt.scatter(col5 , col4 , marker = '*' , color = 'orange')

#plt.scatter(col1 , col4 ,marker = '.' , color = 'green')

#plt.scatter(col2 , col1,marker='*' , color = 'blue')
#plt.scatter(col2 , col4 ,marker = 'v' , color = 'red')
#plt.show()
#dcent_color = [idg[i] for i in range(len(idg))]

#nx.draw(G,pos=None,with_labels=False,node_size=5,node_color=dcent_color,width=0.05)
#plt.savefig("sw0.png",dpi=500)

#plt.plot(col1 , col3)
#plt.show()



#all_nodes = [(node,closeness_centrality(node)) for node in cc]

#top_100_nodes = [n for (n,c) in all_nodes if c in np.argsort(c)[-100:]]

#G1 = G.subgraph(top_100_nodes)  

#top_100_centrality = nx.closeness_centrality(G1)

#nx.draw_spring(G1, k =1, node_color = 'green', \
               #node_size = [top_100_centrality(n) for n in G1.nodes()], 
               #font_size = 6, with_labels = True)
#N_top=10
#colors_top_10=['tab:orange','tab:blue','tab:green','lightsteelblue']
#keys_deg_top=list(deg_centrality_sorted)[0:N_top]
#keys_bet_top=list(betweeness_centrality_sorted)[0:N_top]

#Computing centrality and betweeness intersection
#inter_list=list(set(keys_deg_top) & set(keys_bet_top))

#Setting up color for nodes
#for i in inter_list:
#  color_list[i]=colors_top_10[2]

#for i in range(N_top):
#  if keys_deg_top[i] not in inter_list:
#    color_list[i]=colors_top_10[0]
#    i+=1
#    if(i>=N_top):
#        break
#  if keys_bet_top[i] not in inter_list:
#    color_list[i]=colors_top_10[1]
#    i+=1
#    if(i>=N_top) :
#        break

#Draw graph
#pos= nx.circular_layout(G)
#nx.draw_networkx(G,nx.get_node_attributes(pos, 'text'),with_labels=True,node_color=color_list)

#Setting up legend
#labels=['Top 10 deg cent','Top 10 bet cent','Top 10 deg and bet cent','no top 10']
#for i in range(len(labels)):
#  plt.scatter([],[],label=labels[i],color=colors_top_10[i])
#plt.legend(loc='center')
#plt.show()

#nx.draw_networkx(nx.relabel_nodes(G, nx.betweenness_centrality(G)), with_labels=True, node_color = 'orange')



#nx.draw_networkx(nx.relabel_nodes(G, nx.get_node_attributes(G, 'text')), with_labels=True, node_color = 'orange')
#plot_degree_dist(G)



#plt.show()
    
#for i in result.values():
    #f.write(i[0] + "\n")




junctions_emissions = []
junctions = []
steps=[] 
#emissions_timestamp = [] 
net = None
def get_emmision_at(junction):
    tc2e = 0
    tce = 0
    thce = 0
    tnoe = 0
    tpme = 0
    incoming_edges = junction.getIncoming()
    outgoing_edges = junction.getOutgoing()
    edges = incoming_edges+outgoing_edges
    for edge in edges:
        lanes = edge.getLanes()
        for lane in lanes:
            lane_id = lane.getID()
            co2 = traci.lane.getCO2Emission(lane_id)
            co = traci.lane.getCOEmission(lane_id) 
            hc = traci.lane.getHCEmission(lane_id)
            nox = traci.lane.getNOxEmission(lane_id)
            pmx = traci.lane.getPMxEmission(lane_id)
            tc2e+=co2
            tce+=co
            thce+=hc
            tnoe+=nox
            tpme+=pmx
    return [tc2e,tce,thce,tnoe,tpme]

def compute_dataset(idx):
        emissions = zip(*junctions_emissions[idx])
        tuple_emissions = tuple(emissions)
        co2 = tuple_emissions[0]
        co = tuple_emissions[1]
        hc = tuple_emissions[2]
        nox = tuple_emissions[3]
        pmx = tuple_emissions[4]
        df = pd.DataFrame({'steps':steps, 'co2':co2, 'co':co, 'hc':hc, 'nox':nox, 'pmx':pmx})
        #df2 = df['co2'].mean()
        
        junction = junctions[idx]
        #x, y = junction.getCoord()
        #x, y = net.convertXY2LonLat(x, y)
        #x, y = merc(y, x)
        #x_list = [x]
        #y_list = [y]
        #df2 = pd.DataFrame({'x':x_list, 'y':y_list})
        #print(df2)
        return df
        #, ColumnDataSource(df2)


def modify_doc(doc):
    def make_dataset(idx):
        emissions = zip(*junctions_emissions[idx])
        tuple_emissions = tuple(emissions)
        co2 = tuple_emissions[0]
        co = tuple_emissions[1]
        hc = tuple_emissions[2]
        nox = tuple_emissions[3]
        pmx = tuple_emissions[4]
        df = pd.DataFrame({'steps':steps, 'co2':co2, 'co':co, 'hc':hc, 'nox':nox, 'pmx':pmx})
        
        junction = junctions[idx]
        x, y = junction.getCoord()
        x, y = net.convertXY2LonLat(x, y)
        x, y = merc(y, x)
        x_list = [x]
        y_list = [y]
        df2 = pd.DataFrame({'x':x_list, 'y':y_list})
        return ColumnDataSource(df), ColumnDataSource(df2) 

    def make_plot(src, src2):
        p = figure(
            title='Junction pollution',
            sizing_mode="stretch_width",
            max_width = 1500,
            height = 750
        )
        p.line(source = src, x='steps', y='co2', line_width=2, color='red', legend_label='co2')
        p.line(source = src, x='steps', y='co', line_width=2, color='green', legend_label='co')
        p.line(source = src, x='steps', y='hc', line_width=2, color='blue', legend_label='hc')
        p.line(source = src, x='steps', y='nox', line_width=2, color='orange', legend_label='nox')
        p.line(source = src, x='steps', y='pmx', line_width=2, color='cyan', legend_label='pmx')
        p.legend.location = "top_left"
        p.legend.click_policy="hide" 
        tile_provider = get_provider(OSM)
        x1,y1,x2,y2 = net.getBoundary()
        x1,y1 = net.convertXY2LonLat(x1, y1)
        x2,y2 = net.convertXY2LonLat(x2, y2)
        x1,y1 = merc(y1, x1)
        x2,y2 = merc(y2, x2)
        p1 = figure(x_range=(x1, x2), y_range=(y1, y2), x_axis_type="mercator", y_axis_type="mercator")
        p1.add_tile(tile_provider)
        p1.circle(x='x', y='y', size=12, fill_color="blue", fill_alpha=0.8, source = src2)
        return p, p1

    def update(attr, old, new):
        junction = select.value
        junction_idxs = [junction.getID() for junction in junctions]
        idx = junction_idxs.index(junction)
        new_src, new_src2 = make_dataset(idx)
        src.data.update(new_src.data)
        src2.data.update(new_src2.data)

    
    def merc(lat, lon):
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x/lon
        y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale
        return (x, y)

    select = Select(title="Junctions",  options=[junction.getID() for junction in junctions])
    select.on_change('value',update)
    controls = WidgetBox(select)
    src, src2 = make_dataset(0)
    idx = 0
    # x, y = net.BBoxXY()[0]
    # lat, lon = net.convertXY2LonLat(x, y)
    p, p1 = make_plot(src, src2)
    layout = row(controls, p, p1)
    tab = Panel(child=layout, title = 'Traffic Pollutants')
    tabs = Tabs(tabs=[tab])
    doc.add_root(tabs)

def merc(lat, lon):
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale
    return (x, y)

def run():
    step = 0
    for node in net.getNodes():
        junctions.append(node)
        x, y = node.getCoord()
    for i in range(len(junctions)):
        junctions_emissions.append([])
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if step%1000 == 0:
            for idx, junction in enumerate(junctions):
                emission = get_emmision_at(junction)
                junctions_emissions[idx].append(emission)
                #print(idx , junctions_emissions[idx])
            steps.append(step)
        step += 1
        #print(compute_dataset(240))

    

    
    
        
    traci.close()

    df_e = pd.DataFrame(columns=['id', 'co2', 'co', 'hc', 'nox', 'pmx']) #gives dataframe of emissions at each node

    for id in range(len(junctions)) :
        
        dfz = compute_dataset(id)
        #print(dfz)
        co2 = dfz['co2'].mean()
        co = dfz['co'].mean()
        hc = dfz['hc'].mean()
        nox = dfz['nox'].mean()
        pmx = dfz['pmx'].mean()

        df_e.loc[id] = [id, co2, co, hc, nox, pmx]

    df_e = df_e.iloc[:-14]

    row_sums = df_e.sum(axis=1)
    max_row_index = row_sums.idxmax()
    junction = junctions[max_row_index]
    x, y = junction.getCoord()
    #x, y = net.convertXY2LonLat(x, y)
    #x, y = merc(y, x)
    #print(df_e)



    m = folium.Map(location=[x,y]) #To get the actual location in map
    m.save("index.html")



    result = pd.concat([df, df_e], axis=1)
    print(result)

    #result.plot(x='id', y='co', kind='line')

    col1 = result['id']
    col2 = result['co2']
    col3 = result['co']
    col4 = result['hc']
    col5 = result['nox']
    col6 = result['pmx']
    col7 = result['degree']
    col8 = result['betweeness']
    col9 = result['closeness']
    plt.scatter(col1 , col7 , marker='*' , color = 'orange' )
    plt.scatter(col1 , col6 , marker = '.' , color = 'green')
    plt.show()

    print("Correlations:\n")
    #print("{}".format(df["in degree"].corr(df["out degree"])))

    print("correlation between closeness and pmx emission:{}\n".format(result["closeness"].corr(result["pmx"])))
    print("correlation between betweeness and pmx:{}\n".format(result["betweeness"].corr(result["pmx"])))
    print("correlation between degree and pmx:{}\n".format(result["degree"].corr(result["pmx"])))
    print("correlation between closeness and pmx emission:{}\n".format(result["closeness"].corr(result["co2"])))
    print("correlation between betweeness and pmx:{}\n".format(result["betweeness"].corr(result["co2"])))
    print("correlation between degree and pmx:{}\n".format(result["degree"].corr(result["co2"])))
    print("correlation between closeness and pmx emission:{}\n".format(result["closeness"].corr(result["co"])))
    print("correlation between betweeness and pmx:{}\n".format(result["betweeness"].corr(result["co"])))
    print("correlation between degree and pmx:{}\n".format(result["degree"].corr(result["co"])))
    print("correlation between closeness and pmx emission:{}\n".format(result["closeness"].corr(result["hc"])))
    print("correlation between betweeness and pmx:{}\n".format(result["betweeness"].corr(result["hc"])))
    print("correlation between degree and pmx:{}\n".format(result["degree"].corr(result["hc"])))
    print("correlation between closeness and pmx emission:{}\n".format(result["closeness"].corr(result["nox"])))
    print("correlation between betweeness and pmx:{}\n".format(result["betweeness"].corr(result["nox"])))
    print("correlation between degree and pmx:{}\n".format(result["degree"].corr(result["nox"])))
    #print("correlation between out degree and closeness centrality measures:{}\n".format(df["out degree"].corr(df["hc"])))
    #print("correlation between out degree and betweeness centrality measures:{}\n".format(df["out degree"].corr(df["nox"])))
    #print("correlation between betweeness and closeness centrality measures:{}\n".format(df["betweeness"].corr(df["pmx"])))


    #df_full = pd.DataFrame({'nodes': nodes, 'in degree': in_degree, "out degree": out_degree, "closeness": closeness, 'betweeness': bet_centrality})

    #l = [co2 , co , hc , nox , pmx]
        #print(co2)
        #print(co)
        #print(hc)
        #print(nox)
        #print(pmx)
    #dfc = pd.DataFrame.from_dict({ 'node': junctions , 'Co2': junctions_emissions[0] })
    #print(dfc)

    apps = {'/': Application(FunctionHandler(modify_doc))}

    server = Server(apps, port=5000)
    server.start()
    server.io_loop.start()
    # sys.stdout.flush()


if __name__ == "__main__":
    options = get_options()
    #config = configparser.ConfigParser()
    #config.read('settings.ini')
    #path = config['DEFAULT']['path']
    #out_path = config['DEFAULT']['out_path']
    net = sumolib.net.readNet('/Users/karthikmac/Downloads/ATL/traffic/osm.net.xml')

    if options.nogui:
        sumoBinary = checkBinary('/Users/karthikmac/Downloads/ATL/high_traffic/sumo/bin/sumo')
    else:
        sumoBinary = checkBinary('/Users/karthikmac/Downloads/ATL/high_traffic/sumo/bin/sumo-gui')

    traci.start([sumoBinary, "-c", "/Users/karthikmac/Downloads/ATL/traffic/osm.sumocfg",
                             "--tripinfo-output", "tripinfo.xml", "--time-to-teleport", "-1"])
    run()