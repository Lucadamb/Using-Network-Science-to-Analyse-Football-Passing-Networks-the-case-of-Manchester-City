import networkx as nx
import statistics
from collections import defaultdict
import numpy as np
import warnings
from numpy import var
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings('ignore')

ACCURATE_PASS = 1801

# loading the events data
events = {}
nations = ['England']
for nation in nations:
    with open('./data/events/events_%s.json' % nation) as json_data:
        events[nation] = json.load(json_data)

# loading the match data
matches = {}
nations = ['England']
for nation in nations:
    with open('./data/matches/matches_%s.json' % nation) as json_data:
        matches[nation] = json.load(json_data)

# loading the players data
players = {}
with open('./data/players.json') as json_data:
    players = json.load(json_data)

# loading the competitions data
competitions = {}
with open('./data/competitions.json') as json_data:
    competitions = json.load(json_data)

# loading the competitions data
teams = {}
with open('./data/teams.json') as json_data:
    teams = json.load(json_data)

#get list id_match_for team id
def match_list(team_Id):
    for nation in nations:
        for match in matches[nation]:
            for match_team_Id in match['teamsData']:
                if team_Id == match_team_Id:
                    list_match_wyId.append(match['wyId'])
    return list_match_wyId

#get list of player_for team id
def player_list(team_Id):
    for player in players:
            if player['currentTeamId'] == team_Id:
                player_id = player['wyId']
                player_short_name = player['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                player_team = [player_id, player_short_name]
                list_player.append(player_team)
    return list_player

#function for generete passing network for a match
def passing_networks(match_id):
    # take the names of the two teams of the match
    for nation in nations:
        for match in matches[nation]:
            if match['wyId'] == match_id:
                match_result = match['label']
                print(match['label'])
                for competition in competitions:
                    if competition['wyId'] == match['competitionId']:
                        competition_name = competition['area']['name']
                if(match['label'].split('-')[0].split(' ')[0] == "Manchester" or
                        match['label'].split('-')[0].split(' ')[0] == "West" or
                        match['label'].split('-')[0].split(' ')[0] == "Brighton" or
                        match['label'].split('-')[0].split(' ')[0] == "Tottenham" or
                        match['label'].split('-')[0].split(' ')[0] == "Newcastle" or
                        match['label'].split('-')[0].split(' ')[0] == "AFC" or
                        match['label'].split('-')[0].split(' ')[0] == "Leicester" or
                        match['label'].split('-')[0].split(' ')[0] == "Huddersfield" or
                        match['label'].split('-')[0].split(' ')[0] == "Swansea" or
                        match['label'].split('-')[0].split(' ')[0] == "Stoke" or
                        match['label'].split('-')[0].split(' ')[0] == "Crystal"):
                    team1_name = match['label'].split(' -')[0]
                    team2_name = match['label'].split('- ')[1].split(' ,')[0]

                else:
                    team1_name = match['label'].split('-')[0].split(' ')[0]
                    team2_name = match['label'].split('- ')[1].split(' ,')[0]

    # take the events Pass of the match
    match_events = []
    for ev_match in events[competition_name]:
        if ev_match['matchId'] == match_id:
            if ev_match['eventName'] == 'Pass':
                match_events.append(ev_match)


    team2pass2weight = defaultdict(lambda: defaultdict(int))
    for event, next_event in zip(match_events, match_events[1:]):
        try:
            if event['eventName'] == 'Pass' and ACCURATE_PASS in [tag['id'] for tag in event['tags']]:
                for player in players:
                    if player['wyId'] == event['playerId']:
                        sender = player['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                if next_event['teamId'] == event['teamId']:
                    for player in players:
                        if player['wyId'] == next_event['playerId']:
                            receiver = player['shortName'].encode('ascii', 'strict').decode('unicode-escape')
                            for team in teams:
                                if team['wyId'] == next_event['teamId']:
                                    team2pass2weight[team['name']][(sender, receiver)] += 1
        except KeyError:
            pass


    # crete networkx graphs
    list_weight = []
    G1, G2 = nx.DiGraph(name=team1_name), nx.DiGraph(name=team2_name)
    for (sender, receiver), weight in team2pass2weight[team1_name].items():
        G1.add_edge(sender, receiver, weight=weight)
    for (sender, receiver), weight in team2pass2weight[team2_name].items():
        list_weight.append(weight)
        G2.add_edge(sender, receiver, weight=weight)
    list_pass.append(sum(list_weight))

    return G1, G2, match_result

#function to calculate closeness centrality and betweeness centrality for a player
def get_players_centrality(player_id,player_name):
    #get list of match
    match_list_current_team = []
    for nation in nations:
        for ev in events[nation]:
            if ev['teamId'] == 1625 and ev['playerId'] == player_id:
                if ev['matchId'] not in match_list_current_team:
                    match_list_current_team.append(ev['matchId'])
    list_match_ev = []
    for nation in nations:
        for ev in events[nation]:
            if ev['matchId'] in match_list_current_team:
                list_match_ev.append(ev)

    for match in tqdm(match_list_current_team):
        G1, G2, match_result = passing_networks(match_id = match)
        if(G1.name == "Manchester City"):
            try:
                centrality_closeness = nx.closeness_centrality(G1)[player_name]
                player2centralities_closeness[player_name].append(centrality_closeness)
                centrality_betweenness = nx.betweenness_centrality(G1)[player_name]
                player2centralities_betweenness[player_name].append(centrality_betweenness)
            except KeyError:
                pass
        else:
            try:
                centrality_closeness = nx.closeness_centrality(G2)[player_name]
                player2centralities_closeness[player_name].append(centrality_closeness)
                centrality_betweenness = nx.betweenness_centrality(G2)[player_name]
                player2centralities_betweenness[player_name].append(centrality_betweenness)

            except KeyError:
                pass

    return player2centralities_closeness, player2centralities_betweenness

#function plot graph
def plot_passing_networks(G1, G2):

    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)
    nome2degree = dict(G1.degree)
    #edgewidth1 = [d['weight'] for (u, v, d) in G1.edges(data=True)]

    nx.draw(G1, pos=pos1, nodelist=list(nome2degree.keys()),
            node_size=[deg * 50 for deg in nome2degree.values()],
            node_color='red', edge_color='black',
            with_labels=True, font_weight='bold', alpha=0.75)
    nx.write_gexf(G1, "G1.gexf")
    plt.show()

    nome2degree = dict(G2.degree)
    #edgewidth2 = [d['weight'] for (u, v, d) in G2.edges(data=True)]

    nx.draw(G2, pos=pos2, nodelist=list(nome2degree.keys()),
            node_size=[deg * 50 for deg in nome2degree.values()],
            node_color='blue', edge_color='black',
            with_labels=True, font_weight='bold', alpha=0.75)
    nx.write_gexf(G2, "G2.gexf")
    plt.show()

#function plot centrality
def plot_centrality(players_centralities, names):
    sns.set_style('ticks')

    f, ax = plt.subplots(figsize=(10, 5))
    for player_centralities, player_name in zip(players_centralities, names):
        sns.distplot(pd.DataFrame(player_centralities, columns=['centrality'])['centrality'],
                     label=player_name)
    plt.grid(alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('centrality', fontsize=25)
    plt.ylabel('frequency', fontsize=25)
    lab = ax.legend(loc=1, fontsize=18, frameon=True, shadow=True)
    f.tight_layout()
    plt.show()

#function for calculate metrics for sna in digraph
def social_network_analysis_digraph(G, match_result, match_id):

    G.remove_edges_from(nx.selfloop_edges(G))

    G.remove_node('Bernardo Silva')
    G.remove_node('İ. Gündoğan')
    G.remove_node('Y. Touré')

    info_graph = nx.info(G)
    print('info G')
    print(info_graph)


    #edges
    print('edges analysis')
    # weight
    print(G.edges.data())
    number_of_edge = G.number_of_edges()
    print(number_of_edge)
    list_num_edge_G.append(number_of_edge)

    #degree
    print('degree')
    degree_team = dict(G.degree)
    print(degree_team)
    highest = max(degree_team.values())
    smallest = min(degree_team.values())

    degree = sorted([d for n, d in G.degree()], reverse=True)

    avarage_degree = statistics.mean(degree)
    print("Average degree")
    print("%.2f" % avarage_degree)
    list_degree_avg.append(avarage_degree)

    for player, value in degree_team.items():
        if value == highest:
            listplayer_max_degree.append([player, value])
        if value == smallest:
            listplayer_min_degree.append([player, value])


    #in_degree
    print('In_degree')
    in_degree_team = dict(G.in_degree)
    print(in_degree_team)
    highest_in_degree = max(in_degree_team.values())
    smallest_in_degree = min(in_degree_team.values())
    #variance
    var_in_degree = var(list(in_degree_team.values()))
    print(var_in_degree)

    for player, value in in_degree_team.items():
        if value == highest_in_degree:
            listplayer_max_in_degree.append([player, value])
        if value == smallest_in_degree:
            listplayer_min_in_degree.append([player, value])

    #out_degree
    print('Out_degree')
    out_degree_team = dict(G.out_degree)
    print(out_degree_team)
    highest_out_degree = max(out_degree_team.values())
    smallest_out_degree = min(out_degree_team.values())
    var_out_degree = var(list(out_degree_team.values()))
    print(var_out_degree)

    for player, value in out_degree_team.items():
        if value == highest_out_degree:
            listplayer_max_out_degree.append([player, value])
        if value == smallest_out_degree:
            listplayer_min_out_degree.append([player, value])

    #density
    density = nx.density(G)
    print('density')
    print("%.2f" % density)
    list_density.append(density)

    #edge_connetivity
    edge_connectivity = nx.edge_connectivity(G)
    print('edge_connectivity')
    print(nx.edge_connectivity(G))
    list_edge_connectivity_avg.append(edge_connectivity)

    # Centrality Degree
    degree_centrality = dict(nx.degree_centrality(G))
    ('degree_centrality')
    print(degree_centrality)
    highest_degree_centrality = max(degree_centrality.values())
    smallest_degree_centrality = min(degree_centrality.values())

    for player, value in degree_centrality.items():
        if value == highest_degree_centrality:
            listplayer_max_centrality_degree.append([player, value])
        if value == smallest_degree_centrality:
            listplayer_min_centrality_degree.append([player, value])

    #avarage_degree_centrality
    avarage_degree_centrality = statistics.mean(degree_centrality.values())
    print("average degree centrality")
    print("%.2f" % avarage_degree_centrality)

    #in_degree_centrality
    in_degree_centrality = dict(nx.in_degree_centrality(G))
    print('in_degree_centrality')
    print(in_degree_centrality)
    variance_indegree = var(list(in_degree_centrality.values()))
    print(variance_indegree)
    highest_indegree_centrality = max(in_degree_centrality.values())
    smallest_indegree_centrality = min(in_degree_centrality.values())

    for player, value in in_degree_centrality.items():
        if value == highest_indegree_centrality:
            listplayer_max_centrality_indegree.append([player, value])
        if value == smallest_indegree_centrality:
            listplayer_min_centrality_indegree.append([player, value])

    #out_degree_centrality
    out_degree_centrality = dict(nx.out_degree_centrality(G))
    print('out_degree_centrality')
    print(out_degree_centrality)
    highest_outdegree_centrality = max(out_degree_centrality.values())
    smallest_outdegree_centrality = min(out_degree_centrality.values())
    variance_outdegree = var(list(out_degree_centrality.values()))
    print(variance_outdegree)
    for player, value in out_degree_centrality.items():
        if value == highest_outdegree_centrality:
            listplayer_max_centrality_outdegree.append([player, value])
        if value == smallest_outdegree_centrality:
            listplayer_min_centrality_outdegree.append([player, value])

    print(nx.node_connectivity(G))
    #closeness
    closeness_centrality = dict(nx.closeness_centrality(G))
    print('closeness_centrality')
    print(closeness_centrality)
    #var closeness
    print('variance closeness')
    variance_closeness = var(list(closeness_centrality.values()))
    print(variance_closeness)
    # mean closeness
    print('closeness avg')
    avg_closeness = statistics.mean(list(closeness_centrality.values()))
    print(avg_closeness)
    highest_closeness_centrality = max(closeness_centrality.values())
    smallest_closeness_centrality = min(closeness_centrality.values())

    for player, value in closeness_centrality.items():
        if value == highest_closeness_centrality:
            listplayer_max_closeness_centrality.append([player, value])
        if value == smallest_closeness_centrality:
            listplayer_min_closeness_centrality.append([player, value])

    #betweenness
    betweenness_centrality = dict(nx.betweenness_centrality(G))
    print('betweenness_centrality')
    print(betweenness_centrality)
    avg_b = statistics.mean(list(betweenness_centrality.values()))
    print(avg_b)
    highest_betweenness_centrality = max(betweenness_centrality.values())
    smallest_betweenness_centrality = min(betweenness_centrality.values())

    for player, value in betweenness_centrality.items():
        if value == highest_betweenness_centrality:
            listplayer_max_betweenness_centrality.append([player, value])
        if value == smallest_betweenness_centrality:
            listplayer_min_betweenness_centrality.append([player, value])

    #PageRank
    pagerank = dict(nx.pagerank(G))
    print('pagerank')
    print(pagerank)
    highest_pagerank = max(pagerank.values())
    smallest_pagerank = min(pagerank.values())

    for player, value in pagerank.items():
        if value == highest_pagerank:
            listplayer_max_pagerank.append([player, value])
        if value == smallest_pagerank:
            listplayer_min_pagerank.append([player, value])

    #k-core
    print('max k-core')
    max_kcore = max(nx.core_number(G).values())
    print(max_kcore)

    #clustering coefficient
    clustering_coefficient = dict(nx.clustering(G))
    print('clustering coefficient')
    print(clustering_coefficient)
    print('average clustering')
    avg_clustering = (nx.average_clustering(G))
    print(avg_clustering)
    highest_clustering_coefficient = max(clustering_coefficient.values())
    smallest_clustering_coefficient = min(clustering_coefficient.values())

    for player, value in clustering_coefficient.items():
        if value == highest_clustering_coefficient:
            listplayer_max_clustering_coefficient.append([player, value])
        if value == smallest_clustering_coefficient:
            listplayer_min_clustering_coefficient.append([player, value])

    #clustering coefficient avg
    list_avg_clustering = []
    for item in clustering_coefficient.values():
        list_avg_clustering.append(item)

    avarage_clustering_coefficient = statistics.mean(list_avg_clustering)
    list_clustering_avg.append(avarage_clustering_coefficient)

    print('avarage_clustering coefficient')
    print("%.2f" % avarage_clustering_coefficient)

    print('assortativity')
    assortativity = nx.degree_assortativity_coefficient(G)
    print(assortativity)
    pearson = nx.degree_pearson_correlation_coefficient(G)
    print(pearson)
    # function to add to JSON
    data_match = {'match_id': match_id,
                  'label_match': match_result,
                  'info': info_graph,
                  'density': density,
                  'degree': degree_team,
                  'avarage_degree': avarage_degree,
                  'in_degree': in_degree_team,
                  'out_degree': out_degree_team,
                  'degree_centrality': degree_centrality,
                  'avarage_degree_centrality': avarage_degree_centrality,
                  'in_degree_centrality': in_degree_centrality,
                  'out_degree_centrality': out_degree_centrality,
                  'closeness_centrality': closeness_centrality,
                  'betweenness_centrality': betweenness_centrality,
                  'pagerank': pagerank,
                  'edge_connectivity': edge_connectivity,
                  'max-k-core': max_kcore,
                  'clustering_coefficient': clustering_coefficient,
                  'avarage_clustering_coefficient': avarage_clustering_coefficient,
                  }
    with open('sna_match_'+str(match_id)+'2.json', 'w') as f:
        json.dump(data_match, f, indent=4)

    return list_num_edge_G

#function for calculate metrics for sna in graph
def social_network_analysis_graph(G, match):

    #function for add JSON code
    def write_json(data, filename='sna_match_' + str(match) + '2.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    if G.number_of_nodes() > 0:
        G = G.to_undirected()
        if nx.is_connected(G):
            clique = nx.find_cliques(G)
            for x in clique:
                print(x)
            from networkx.algorithms.approximation import clique
            max_clique = nx.graph_number_of_cliques(G)
            print('')
            print("Maximum clique number")
            print(max_clique)
            list_max_clique.append(max_clique)
            #triangles
            triangles = nx.triangles(G)
            print("Triangles in team")
            print(triangles)
            num_triangles = (sum(list(nx.triangles(G).values())))
            print("Number of triangles")
            print(num_triangles)
            #transitivity
            transitivity = nx.transitivity(G)
            print("Transitivity")
            print("%.2f" % transitivity)
            list_transitivity_avg.append(transitivity)

            with open('sna_match_' + str(match) + '.json') as json_file:
                data = json.load(json_file)

                data_match = {
                    'max clique number': max_clique,
                    'triangles for node': triangles,
                    'number_of_triangles': num_triangles,
                    'transitivity': transitivity,
                }
                # appending data
                data.update(data_match)
            write_json(data)

#function for calculate avg value order max
def statistical_analysis_season_max(list):
    dict = {}
    for elem in list:
        if elem[0] not in dict:
            dict[elem[0]] = []
        dict[elem[0]].append(elem[1:])

    for key in dict:
        for i in zip(*dict[key]):
            dict[key] = (sum(i)/38)

    sort_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    return sort_list

#function for calculate avg value order min
def statistical_analysis_season_min(list):
    dict = {}
    for elem in list:
        if elem[0] not in dict:
            dict[elem[0]] = []
        dict[elem[0]].append(elem[1:])

    for key in dict:
        for i in zip(*dict[key]):
            dict[key] = (sum(i)/38)

    sort_list = sorted(dict.items(), key=lambda x: x[1])

    return sort_list

#Passing network
list_match_wyId = []
list_num_edge = []
list_num_edge1 = []
list_num_edge_G = []
list_pass = []
list_density = []

#avg final in/out/degree
list_degree_avg = []

#avg final edge
list_edge_connectivity_avg = []

#avg clustering
list_clustering_avg = []
list_transitivity_avg =  []
list_max_clique = []
list_betweenness_centrality = []


#PLAYER LIST
#MaxMin in/ou/degree
listplayer_max_in_degree = []
listplayer_max_out_degree = []
listplayer_max_degree = []
listplayer_min_in_degree = []
listplayer_min_out_degree = []
listplayer_min_degree = []


#MaxMin list centrality in/out/degree
listplayer_max_centrality_degree = []
listplayer_max_centrality_outdegree = []
listplayer_max_centrality_indegree = []
listplayer_min_centrality_degree = []
listplayer_min_centrality_outdegree = []
listplayer_min_centrality_indegree = []

#MaxMin list closeness centrality
listplayer_max_closeness_centrality = []
listplayer_min_closeness_centrality = []

#MaxMin list betweenness_centrality
listplayer_max_betweenness_centrality = []
listplayer_min_betweenness_centrality = []

#MaxMin clustering_coefficient
listplayer_max_clustering_coefficient = []
listplayer_min_clustering_coefficient = []

#MaxMin pagerank

listplayer_max_pagerank = []
listplayer_min_pagerank = list ()


match = match_list(team_Id='1625')

for match_id in match:
    print(match_id)
    G1, G2, match_result = passing_networks(match_id=match_id)
    #plot_passing_networks(G1, G2)
    if (G1.name == "Manchester City"):
        list_num_edge = social_network_analysis_digraph(G1, match_result,match_id)
        social_network_analysis_graph(G1, match_id)
    else:
        list_num_edge1 = social_network_analysis_digraph(G2, match_result, match_id)
        social_network_analysis_graph(G2, match_id)

#ManCity - Stoke

G1, G2, match_result = passing_networks(match_id=2499794)
social_network_analysis_digraph(G1, match_result, 2499794)
social_network_analysis_graph(G1, 2499794)
plot_passing_networks(G1, G2)


print('\nmax_degree_centrality_avg')
max_degree_centrality_avg = statistical_analysis_season_max(listplayer_max_centrality_degree)
print('\n max_in_degree_centrality_avg')
max_in_degree_centrality_avg = statistical_analysis_season_max(listplayer_max_centrality_indegree)
print('\n max_out_degree_cetrality_avg')
max_out_degree_centrality_avg = statistical_analysis_season_max(listplayer_max_centrality_outdegree)

print('\nmin_degree_centrality_avg')
min_degree_centrality_avg = statistical_analysis_season_min(listplayer_min_centrality_degree)
print('\n min_indegree_centrality_avg')
min_indegree_centrality_avg = statistical_analysis_season_min(listplayer_min_centrality_indegree)
print('\n min_outdegree_centrality_avg')
min_outdegree_centrality_avg = statistical_analysis_season_min(listplayer_min_centrality_outdegree)

print('\nlistplayer_max_closeness_centrality')
max_closeness_centrality_avg = statistical_analysis_season_max(listplayer_max_closeness_centrality)

print('\n listplayer_min_closeness_centrality')
min_closeness_centrality_avg = statistical_analysis_season_min(listplayer_min_closeness_centrality)
for element in listplayer_min_betweenness_centrality:
    print(element[0])
    if element[1] == 0.0:
        listplayer_min_betweenness_centrality.remove(element)
        v = -1
        listplayer_min_betweenness_centrality.append([element[0],v])
print('\nlistplayer_max_betweenness_centrality')
max_betweenness_centrality_avg = statistical_analysis_season_max(listplayer_max_betweenness_centrality)
print('\n listplayer_min_betweenness_centrality')
min_betweenness_centrality_avg = statistical_analysis_season_min(listplayer_min_betweenness_centrality)

print('\nlistplayer_max_clustering_coefficient')
max_clustering_coefficient_avg = statistical_analysis_season_max(listplayer_max_clustering_coefficient)
print('\n listplayer_min_betweenness_centrality')
min_clustering_coefficient_avg = statistical_analysis_season_min(listplayer_min_clustering_coefficient)

print('\nlistplayer_max_pagerank')
max_pagerank_avg = statistical_analysis_season_max(listplayer_max_pagerank)
print('\n listplayer_min_pagerank')
min_pagerank_avg = statistical_analysis_season_min(listplayer_min_pagerank)


data_season = {
                'max_degree_centrality_avg': max_degree_centrality_avg,
                'max_indegree_centrality_avg': max_in_degree_centrality_avg,
                'max_outdegree_centrality_avg': max_out_degree_centrality_avg,

                'min_degree_centrality_avg': min_degree_centrality_avg,
                'min_indegree_centrality_avg': min_indegree_centrality_avg,
                'min_outdegree_centrality_avg': min_outdegree_centrality_avg,

                'max_closeness_centrality_avg': max_closeness_centrality_avg,
                'min_closeness_centrality_avg': min_closeness_centrality_avg,

                'max_betweenness_centrality_avg':max_betweenness_centrality_avg,
                'min_betweenness_centrality_avg':min_betweenness_centrality_avg,

                'max_clustering_coefficient_avg': max_clustering_coefficient_avg,
                'min_clustering_coefficient_avg': min_clustering_coefficient_avg,

                'max_pagerank_avg': max_pagerank_avg,
                'min_pagerank_avg': min_pagerank_avg
}

with open('sna_season_player_manchester_city.json', 'w') as f:
    json.dump(data_season, f, indent=4)


print('avg density')
avg_density = statistics.mean(list_density)
print(avg_density)

print('avg degree')
avg_degree = statistics.mean(list_degree_avg)
print(avg_degree)

print('edge connectivity avg')
avg_edge_connectivity = statistics.mean(list_edge_connectivity_avg)
print(avg_edge_connectivity)

print(' list_clustering_avg ')
avg_clustering_global = statistics.mean(list_clustering_avg)
print(avg_clustering_global)

print(' max_clique ')
avg_max_clique = statistics.mean(list_max_clique)
print(avg_max_clique)

print(' transivity ')
avg_transivity = statistics.mean(list_transitivity_avg)
print(avg_transivity)


list_num_edge_2 = list_num_edge1 + list_num_edge
avg_edge_G = statistics.mean(list_num_edge_2)
max_edge = max(list_num_edge_2)
min_edge = min(list_num_edge_2)
print("Statistics passing")
print("avg: %.2f" % avg_edge_G, 'min:', min_edge, 'max:',max_edge)

list_player = []
player = player_list(team_Id=1625)
#centrality

player2centralities_closeness = defaultdict(list)
player2centralities_betweenness = defaultdict(list)

for name_id_player in player:
    print(name_id_player[0])
    print(name_id_player[1])
    player2centralities_closeness, player2centralities_betweenness = get_players_centrality(name_id_player[0], name_id_player[1])

#debruyne_fcs = player2centralities_closeness['K. De Bruyne']
stones_fc = player2centralities_closeness['J. Stones']
fernandinho_fc = player2centralities_closeness['Fernandinho']
ederson_fc = player2centralities_closeness['Ederson']
aguero_fc = player2centralities_closeness['S. Agüero']

mean_closeness_stones = statistics.mean(stones_fc)
mean_closeness_fernandinho_fc = statistics.mean(fernandinho_fc)
mean_closeness_ederson = statistics.mean(ederson_fc)
mean_closeness_aguero = statistics.mean(aguero_fc)
print("Stones mean: ", mean_closeness_stones)
print("Fernandinho mean: ", mean_closeness_fernandinho_fc)
print("Ederson mean: ", mean_closeness_ederson)
print("Aguero mean: ", mean_closeness_aguero)

stones_fb = player2centralities_betweenness['J. Stones']
fernandinho_fb = player2centralities_betweenness['Fernandinho']
ederson_fb = player2centralities_betweenness['Ederson']
aguero_fb = player2centralities_betweenness['S. Agüero']

mean_betweenness_stones = statistics.mean(stones_fb)
mean_betweenness_fernandinho = statistics.mean(fernandinho_fb)
mean_betweenness_ederson = statistics.mean(ederson_fb)
mean_betweenness_aguero = statistics.mean(aguero_fb)
print("Stones mean b: ", mean_betweenness_stones)
print("Fernandinho meanb : ", mean_betweenness_fernandinho)
print("Ederson mean b: ", mean_betweenness_ederson)
print("Aguero mean b: ", mean_betweenness_aguero)


#davidsilva_fcs = player2centralities_closeness['David Silva']
#gündoğan_fcs = player2centralities_closeness['İ. Gündoğan']
#foden_fcs = player2centralities_closeness['P. Foden']


plot_centrality([stones_fc, fernandinho_fc, ederson_fc, aguero_fc],
                     ['J. Stones','Fernandinho','Ederson','S. Agüero'])

plot_centrality([stones_fb, fernandinho_fb, ederson_fb, aguero_fb],
                     ['J. Stones','Fernandinho','Ederson','S. Agüero'])

