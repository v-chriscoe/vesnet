from pyvis.network import Network
import pandas as pd
import networkx as nx
from datetime import datetime
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore')


def get_network(df_events, size_metric=None, filename=None, node_of_interest=None):
    name_mmsi = get_name_mmsi(df_events)
    
    edges = get_edges(df_events)
    _,_,namesUnique = get_unique_pairs(df_events)
    
    if size_metric:
        node_size_list = get_node_size(size_metric, namesUnique, edges)
        node_size = node_size_list/max(node_size_list)
    else:
        node_size = 10

    net = Network(
        notebook=True,
        neighborhood_highlight=True,
        cdn_resources="remote",
        bgcolor="#222222",
        font_color="white",
        height="800px",
        width="100%",
        select_menu=True,
        filter_menu=True)
    
     # net.set_template_dir('/lib')
    # net.show_buttons(filter_=['physics']) 
    net.set_template_dir('../templates', template_file='template.html')
    
    nx_graph = get_networkx(edges, name_mmsi)
    
    if node_of_interest:   
        #Node of interest name to mmsi
        node_names_rev = get_node_names(df_events, rev=False)
        # print(node_names_rev)
        node_of_interest_mmsi = node_names_rev.get(node_of_interest)
        # print(node_of_interest_mmsi)
        nx_graph_2_hops = nx.single_source_shortest_path_length(nx_graph, node_of_interest_mmsi, cutoff=2)
        neighbor_list = list(nx_graph_2_hops.keys())
        nx_graph = nx_graph.subgraph(neighbor_list)
    
    node_names = get_node_names(df_events, rev=False)
    nx_graph = nx.relabel_nodes(nx_graph, node_names)
 
    net.from_nx(nx_graph)
   
    if filename:
        net.write_html('../plots/' + filename)
    else:
        net.write_html('../plots/Network.html')

    return net


def get_name_mmsi(df_events):
    df_events['vessels.vessel_0.category'][(df_events['vessels.vessel_0.category'] == 'other') | 
                                           (df_events['vessels.vessel_0.category'] == 'unknown')] = df_events['vessels.vessel_0.subcategory']
    df_events['vessels.vessel_1.category'][(df_events['vessels.vessel_1.category'] == 'other') | 
                                           (df_events['vessels.vessel_1.category'] == 'unknown')] = df_events['vessels.vessel_1.subcategory']
    data = [df_events[['vessels.vessel_0.name', 'vessels.vessel_0.mmsi', 'vessels.vessel_0.category']].rename(columns={'vessels.vessel_0.name':'Vessel_name', 'vessels.vessel_0.mmsi':'MMSI','vessels.vessel_0.category':'Vessel_Type'}), 
    df_events[['vessels.vessel_1.name', 'vessels.vessel_1.mmsi', 'vessels.vessel_1.category']].rename(columns={'vessels.vessel_1.name':'Vessel_name', 'vessels.vessel_1.mmsi':'MMSI', 'vessels.vessel_1.category':'Vessel_Type'})]
    name_mmsi = pd.concat(data, axis=0)

    name_mmsi.Vessel_Type.fillna(value='None', inplace=True)
    name_mmsi['Total_RDV'] = name_mmsi.groupby(['Vessel_name','MMSI','Vessel_Type'])['Vessel_name'].transform('count')

    return name_mmsi


def get_edges(df_events):
    edges=[]
    mmsiPairsUnique,mmsiPairsCount,_ = get_unique_pairs(df_events)

    for mmsiPair, weight in zip(mmsiPairsUnique,mmsiPairsCount):
        mmsi1,mmsi2 = mmsiPair.split(',')
        mmsi1 = int(float(mmsi1))
        mmsi2 = int(float(mmsi2))
        weight = int(weight)
        edge = (mmsi1,mmsi2,weight)
        edges.append(edge)  

    return edges


def get_unique_pairs(df_events):
    
    mmsi1s = []
    mmsi2s = []
    name1s = []
    name2s = []

    # for event in eventList:
    for index, rows in df_events.iterrows():    
        #print(event)

        vesselmmsi0 = rows['vessels.vessel_0.mmsi']
        vesselmmsi1 = rows['vessels.vessel_1.mmsi']
        vesselname0 = rows['vessels.vessel_0.name']
        vesselname1 = rows['vessels.vessel_1.name']

        mmsi1s.append(vesselmmsi0)
        mmsi2s.append(vesselmmsi1)
        name1s.append(vesselname0)
        name2s.append(vesselname1)

    #make sure all vessel pairs in the same order 
    #so duplicates are correctly identified

    for i in range(0,len(mmsi1s)):
        if mmsi2s[i] < mmsi1s[i]:
            mmsi2s[i],mmsi1s[i] = mmsi1s[i],mmsi2s[i]
            name2s[i],name1s[i] = name1s[i],name2s[i]
        else:
            pass
    
    mmsiPairs = []
    mmsisUnique = []
    namesUnique = []
    for mmsi1,mmsi2,name1,name2 in zip(mmsi1s,mmsi2s,name1s,name2s):
        if mmsi1 not in mmsisUnique:
            mmsisUnique.append(mmsi1)
            namesUnique.append(name1)
        else:
            pass
        if mmsi2 not in mmsisUnique:
            mmsisUnique.append(mmsi2) 
            namesUnique.append(name2)
        else:
            pass
        mmsiPairs.append('{},{}'.format(mmsi1,mmsi2))

    #now checking for unique pairs which will becomes edges
    #and counting duplicates which will become weights
    mmsiPairsUnique = []
    mmsiPairsCount = []
    for mmsiPair in mmsiPairs:
        if mmsiPair not in mmsiPairsUnique:
            mmsiPairsUnique.append(mmsiPair)
            mmsiPairsCount.append(1)
        elif mmsiPair in mmsiPairsUnique:
            index = mmsiPairsUnique.index(mmsiPair)
            mmsiPairsCount[index] += 1
        else:
            pass
        
    return mmsiPairsUnique,mmsiPairsCount,namesUnique

def get_networkx(edge_list, name_mmsi):
    # nodes = namesUnique
    edges = edge_list
    # print(edges)
    # Create a new graph
    G = nx.Graph()

    # Add nodes to the graph
    # G.add_nodes_from(nodes)
    
    for index, row in name_mmsi.iterrows():
        G.add_node(row['MMSI'], label=row['Vessel_name'],
                     size=10,
                     title=('MMSI: ' + str(row['MMSI']) 
                            + '\n'
                            'Vessel Name: ' + str(row['Vessel_name']) 
                            + '\n'
                            + 'Vessel Type: '+str(row['Vessel_Type']) 
                            + '\n'
                            + 'Total_RDV: ' + str(row['Total_RDV'])))

    # # Add edges to the graph
    for e in edges:
        G.add_edge(e[0], e[1])
        
    return G


def get_node_size(size_metric, namesUnique, edge_list):
    nodes = namesUnique
    edges = edge_list
    # print(edges)
    # Create a new graph
    G = get_networkx(namesUnique, edge_list)

#     # Add nodes to the graph
#     G.add_nodes_from(nodes)

#     # # Add edges to the graph
#     for u,v in zip(edges[0], edges[1]):
#         G.add_edge(u, v)
    
#     G = get_networkx(namesUnique, edge_list)
    
    cent = []    
    if size_metric=='betweenness_centrality':
        for node in nodes:
            cent.append(nx.betweenness_centrality(G)[node])
    
    elif size_metric=='degree_centrality':
        for node in nodes:
            cent.append(nx.betweenness_centrality(G)[node])
        
    return cent

def get_node_names(df_events, rev=False):
    mmsi1s = []
    mmsi2s = []
    name1s = []
    name2s = []

    # for event in eventList:
    for index, rows in df_events.iterrows():    
        #print(event)

        vesselmmsi0 = rows['vessels.vessel_0.mmsi']
        vesselmmsi1 = rows['vessels.vessel_1.mmsi']
        vesselname0 = rows['vessels.vessel_0.name']
        vesselname1 = rows['vessels.vessel_1.name']

        mmsi1s.append(vesselmmsi0)
        mmsi2s.append(vesselmmsi1)
        name1s.append(vesselname0)
        name2s.append(vesselname1)

    #make sure all vessel pairs in the same order 
    #so duplicates are correctly identified

    for i in range(0,len(mmsi1s)):
        if mmsi2s[i] < mmsi1s[i]:
            mmsi2s[i],mmsi1s[i] = mmsi1s[i],mmsi2s[i]
            name2s[i],name1s[i] = name1s[i],name2s[i]
        else:
            pass

    #now iterating again to join into a list of mmsi pairs
    #and creating the list of unique MMSI's for nodes
    mmsiPairs = []
    mmsisUnique = []
    namesUnique = []
    for mmsi1,mmsi2,name1,name2 in zip(mmsi1s,mmsi2s,name1s,name2s):
        if mmsi1 not in mmsisUnique:
            mmsisUnique.append(mmsi1)
            namesUnique.append(name1)
        else:
            pass
        if mmsi2 not in mmsisUnique:
            mmsisUnique.append(mmsi2) 
            namesUnique.append(name2)
        else:
            pass
        mmsiPairs.append('{},{}'.format(mmsi1,mmsi2))

    #now checking for unique pairs which will becomes edges
    #and counting duplicates which will become weights
    mmsiPairsUnique = []
    mmsiPairsCount = []
    for mmsiPair in mmsiPairs:
        if mmsiPair not in mmsiPairsUnique:
            mmsiPairsUnique.append(mmsiPair)
            mmsiPairsCount.append(1)
        elif mmsiPair in mmsiPairsUnique:
            index = mmsiPairsUnique.index(mmsiPair)
            mmsiPairsCount[index] += 1
        else:
            pass
    
    if rev is True:
        node_names = dict(zip(mmsisUnique, namesUnique))
    else:
        node_names = dict(zip(namesUnique, mmsisUnique))
    return node_names

def get_hops(df_events, node_of_interest, hops, name_mmsi):
    _,_,namesUnique = get_unique_pairs(df_events)
    edge_list = get_edges(df_events)
    node_names = get_node_names(df_events)
    # print(node_names)
    G = get_networkx(edge_list, name_mmsi)
    # Change node label
    G = nx.relabel_nodes(G, node_names)
    G_Sub_nodes = nx.single_source_shortest_path_length(G, node_of_interest, cutoff=hops)
    nbr_list = list(G_Sub_nodes.keys())
    return nbr_list


def get_subgraph(df_events, node_of_interest, cutoff, size_metric=None, filename=None):
    # df_events_subset = df_events[(df_events['vessels.vessel_0.name'].isin(neighbor_list)) | (df_events['vessels.vessel_1.name'].isin(neighbor_list)) 
    #                 |(df_events['vessels.vessel_0.name'].isin(neighbor_list)) | (df_events['vessels.vessel_1.name'].isin(neighbor_list))]
    edges = get_edges(df_events)
    print(edges)
    _,_,namesUnique = get_unique_pairs(df_events)
    nx_graph = get_networkx(namesUnique, edges)
    
    node_mmsi = get_node_names(df_events, rev=True)
    G_Sub_nodes = nx.single_source_shortest_path_length(nx_graph, node_of_interest, cutoff=cutoff)
    neighbor_list = list(G_Sub_nodes.keys())
    
    nx_graph = nx.relabel_nodes(nx_graph, node_mmsi)  
 
    neighbor_list_df = df_events[(df_events['vessels.vessel_0.name'].isin(neighbor_list)) | (df_events['vessels.vessel_1.name'].isin(neighbor_list)) 
                    |(df_events['vessels.vessel_0.name'].isin(neighbor_list)) | (df_events['vessels.vessel_1.name'].isin(neighbor_list))]
    neighbor_list_df = neighbor_list_df[['vessels.vessel_0.mmsi','vessels.vessel_1.mmsi']]
    neighbor_list_mmsi=[]
    for i in neighbor_list_df['vessels.vessel_0.mmsi']:
        neighbor_list_mmsi.append(int(i))
    for i in neighbor_list_df['vessels.vessel_1.mmsi']:
        neighbor_list_mmsi.append(int(i))
    neighbor_list_mmsi = list(set(neighbor_list_mmsi))
    print(neighbor_list_mmsi)
    
    sub = nx_graph.subgraph(neighbor_list_mmsi)
    
    node_names = get_node_names(df_events, rev=False)
    sub = nx.relabel_nodes(sub, node_names)  
    
    sub_net = Network(
        notebook=True,
        neighborhood_highlight=True,
        cdn_resources="remote",
        bgcolor="#222222",
        font_color="white",
        height="800px",
        width="100%",
        select_menu=True,
        filter_menu=True)
    sub_net.from_nx(sub)

    if filename:
        sub_net.write_html(filename)
    else:
        sub_net.write_html('sub_network.html')

    return sub_net