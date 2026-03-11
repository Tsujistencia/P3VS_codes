import dash
import base64
import os
import plotly.express as px
from matplotlib import mlab
import matplotlib
matplotlib.use('Agg')  # GUI不要なバックエンドを使用

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import Levenshtein
import networkx as nx
from sklearn_extra.cluster import KMedoids
from dash import Dash, html, dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
server = app.server

image_directory = '/Users/tsujistencia/Desktop/Cash'
image_extension = '.png'

def select_extraction_data(name, season, Stype, selected_nodes, category):
    path = f'/Users/tsujistencia/Desktop/Player_Data/{name}.csv'

    data = pd.read_csv(path, usecols=['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
                                      'release_pos_y', 'release_pos_z', 'player_name', 'batter',
                                      'pitcher', 'zone', 'stand', 'p_throws', 'type', 'bb_type',
                                      'inning', 'inning_topbot', 'effective_speed', 'release_spin_rate',
                                      'game_pk', 'fielder_2', 'at_bat_number', 'pitch_number', 'pitch_name','events'])

    data['game_date'] = pd.to_datetime(data['game_date'])
    data['year'] = data['game_date'].dt.year
    zone_mapping = {
        1: 'c', 2: 'b', 3: 'a', 4: 'f', 5: 'e', 6: 'd',
        7: 'i', 8: 'h', 9: 'g', 11: 'k', 12: 'j', 13: 'm', 14: 'l'
    }
    data['zone'] = data['zone'].map(zone_mapping).fillna(data['zone'])
    data.sort_values(by=['game_date', 'game_pk', 'inning', 'inning_topbot', 'at_bat_number', 'pitch_number'],
                     ascending=[True, True, True, False, True, True], inplace=True)
    print('抽出前', len(data))
    print(category)
    data = data[data['year'].isin(season)]

    if Stype != 'All':
        data = data[data['stand'] == Stype]

    gb = data.groupby(['game_date', 'inning', 'at_bat_number'])
    trajects = []
    results = []
    for i, g in gb:
        if len(g) == 1:
            continue
        else:
            s = ''.join(map(str, g['zone']))
            trajects.append(s)
            result = g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None
            results.append(result)

    return_df = pd.DataFrame({
        'trajectory': trajects,
        'Result': results,
    })

    event_map = {
        'stolen_base_2b': 'Out', 'catcher_interf': 'Walk',
        'double_play': 'Out', 'field_out': 'Out', 'fielders_choice_out': 'Out', 'force_out': 'Out',
        'grounded_into_double_play': 'Out', 'strikeout': 'StrikeOut',
        'strikeout_double_play': 'StrikeOut',
        'double': 'BaseHit', 'hit_by_pitch': 'Walk', 'single': 'BaseHit', 'triple': 'BaseHit', 'walk': 'Walk',
        'home_run': 'HomeRun',
    }

    return_df['Category'] = return_df['Result'].map(event_map).fillna(return_df['Result'])

    if category == 'All':
        return return_df
    else:
        return_df1 = return_df[return_df['Category'].isin([category])]
        print('抽出後', len(return_df1))

        if len(selected_nodes) == 0:
            return return_df1.reset_index(drop=True)

        filtered_data = return_df1[return_df1['trajectory'].apply(
            lambda item: len(item) >= len(selected_nodes) and all(item[i] == selected_nodes[i] for i in range(len(selected_nodes)))
        )]

        print(len(filtered_data))
        return filtered_data.reset_index(drop=True)

def clustering_data(filtered_data, num_classes):
    num_classes = int(num_classes)
    dist = np.zeros((len(filtered_data), len(filtered_data)))
    for i in range(len(filtered_data)):
        for j in range(len(filtered_data)):
            str1 = filtered_data['trajectory'][i]
            str2 = filtered_data['trajectory'][j]
            d = Levenshtein.distance(str1, str2)
            dist[i][j] = d
            dist[j][i] = d

    kmedoids = KMedoids(n_clusters=num_classes, random_state=0, init='k-medoids++',metric='precomputed')
    kmedoids.fit(dist)
    clustered_data = kmedoids.labels_
    return clustered_data

def kmedoids_fit(data,label):
    dic={
        'trajects':data['trajectory'],
        'Category':data['Category'],
        'class':label,
    }
    kmedoids_results = pd.DataFrame(dic)
    print(kmedoids_results)
    return kmedoids_results

def class_choice(data,num):
    class_data = data[data['class'].isin([int(num)])]
    class_data = class_data['trajects']
    class_data = class_data.values.tolist()
    return class_data

def nodeCount0(data):
    nodeDic = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, 'l': 0, 'm': 0}

    for i in range(len(data)):
        for j in range(len(data[i])):
            node = data[i][j]
            if node in nodeDic:
                nodeDic[node] += 1

    return nodeDic

def edgeCount(data):
    edgeDic = {}
    for i in range(0,len(data)):
        for j in range(1, len(data[i])):
            edge = (data[i][j-1],data[i][j])
            if edge in edgeDic:
                edgeDic[edge] += 1
            else:
                edgeDic[edge] = 1

    return edgeDic

def edge_merge(data):
    result = {}

    for key, value in data:
        normalized_key = tuple(sorted(key))
        if normalized_key in result:
            result[normalized_key] += value
        else:
            result[normalized_key] = value

    return result

def create_NX(clustered_data, pitcher_name, year, stance_type, selected_zones, num_classes, category, nodesize_unit):
    Cash = '/Users/tsujistencia/Desktop/Cash'

    allowed_nodes = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
    s_zone = ['a','b','c','d','e','f','g','h','i']
    b_zone = ['j','k','l','m']

    update_Path = None

    for i in range(0, int(num_classes)):
        G = nx.Graph()

        # ノード登録（色も以前と同じまま）
        G.add_nodes_from([
            ('a', {"color": "darkorange"}),
            ('b', {"color": "darkorange"}),
            ('c', {"color": "darkorange"}),
            ('d', {"color": "darkorange"}),
            ('e', {"color": "darkorange"}),
            ('f', {"color": "darkorange"}),
            ('g', {"color": "darkorange"}),
            ('h', {"color": "darkorange"}),
            ('i', {"color": "darkorange"}),
            ('j', {"color": "skyblue"}),
            ('k', {"color": "skyblue"}),
            ('l', {"color": "skyblue"}),
            ('m', {"color": "skyblue"}),
        ])

        test_data = class_choice(clustered_data, i)
        node00 = nodeCount0(test_data)

        nodesize = []
        for node_id in G.nodes:
            if node_id in allowed_nodes:
                cnt = node00.get(node_id, 1)
                nodesize.append(nodesize_unit * cnt)
            else:
                nodesize.append(nodesize_unit)
       
        edge01 = edgeCount(test_data)
        dic02 = sorted(edge01.items(), key=lambda x: x[1], reverse=True)
        edge_data00 = edge_merge(dic02)

        for k, v in edge_data00.items():
            if v == 1 or k[0] == k[1]:
                continue
            if (k[0] not in allowed_nodes) or (k[1] not in allowed_nodes):
                continue

            if (k[0] in b_zone) & (k[1] in b_zone):
                G.add_edge(k[0], k[1], weight=v, color='b')
            elif (k[0] in s_zone) & (k[1] in s_zone):
                G.add_edge(k[0], k[1], weight=v, color='r')
            else:
                G.add_edge(k[0], k[1], weight=v, color='g')

        node_color = [node["color"] for node in G.nodes.values()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        edge_colors = [G[u][v]['color'] for u, v in G.edges]

        pos = {
            'a':[-1.2,1.5], 'b':[0, 1.9], 'c':[1.2, 1.5],
            'd':[-1.7,0], 'e':[0.1,0.3], 'f':[1.7,0],
            'g':[-1.2,-1.5],'h':[0,-1.9], 'i':[1.2,-1.5],
            'j':[-3,2], 'k':[3,2], 'l':[-3,-2],
            'm':[3,-2]
        }

        update_Path = os.path.join(
            Cash,
            f'{pitcher_name}_{year}_{stance_type}_{category}_{selected_zones}_{num_classes}_{i}.png'
        )

        if os.path.exists(update_Path):
            print(f'File {update_Path} already exists. Skipping plot.')
            continue

        fig = plt.figure()
        nx.draw(
            G, pos, with_labels=True,
            node_size=nodesize,
            node_color=node_color,
            edge_color=edge_colors,
            width=edge_weights
        )

        plt.savefig(update_Path)
        plt.close('all')

    return update_Path

def update_NX(pitcher_name, year, stance_type, selected_zones, num_classes, categories):
    images_html = []
    
    for category in categories:  
        category_images = []  
        
        for i in range(int(num_classes)):
            filename = f"{pitcher_name}_{year}_{stance_type}_{category}_{selected_zones}_{num_classes}_{i}{image_extension}"
            image_path = os.path.join(image_directory, filename)

            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img = base64.b64encode(img_file.read()).decode('utf-8')
                    category_images.append(html.Img(src=f'data:image/png;base64,{img}',
                                                    style={'height': '15%', 'width': '15%', 'object-fit': 'contain'}))
            else:
                category_images.append(html.Div(f"Image not found: {filename}"))

        category_html = html.Div([
            html.H4(f"Category: {category}"),
            html.Div(category_images, style={'display': 'flex', 'flex-direction': 'row', 'gap': '10px','justify-content':'center'})
        ])
        images_html.append(category_html)

    return images_html

# Node positions for cytoscape layout
nodes = [
    {"data": {"id": "a", "label": "A"}, "position": {"x": 200, "y": 200},"locked": True},
    {"data": {"id": "b", "label": "B"}, "position": {"x": 400, "y": 180},"locked": True},
    {"data": {"id": "c", "label": "C"}, "position": {"x": 600, "y": 200},"locked": True},
    {"data": {"id": "d", "label": "D"}, "position": {"x": 180, "y": 400},"locked": True},
    {"data": {"id": "e", "label": "E"}, "position": {"x": 400, "y": 380},"locked": True},
    {"data": {"id": "f", "label": "F"}, "position": {"x": 620, "y": 400},"locked": True},
    {"data": {"id": "g", "label": "G"}, "position": {"x": 200, "y": 600},"locked": True},
    {"data": {"id": "h", "label": "H"}, "position": {"x": 400, "y": 620},"locked": True},
    {"data": {"id": "i", "label": "I"}, "position": {"x": 600, "y": 600},"locked": True},
    {"data": {"id": "j", "label": "J"}, "position": {"x": 0, "y": 0},"locked": True},
    {"data": {"id": "k", "label": "K"}, "position": {"x": 800, "y": 0},"locked": True},
    {"data": {"id": "l", "label": "L"}, "position": {"x": 0, "y": 800},"locked": True},
    {"data": {"id": "m", "label": "M"}, "position": {"x": 800, "y": 800},"locked": True},
]

edges = []
elements_init = nodes + edges

cyto_compo1 = cyto.Cytoscape(
    id="dash_cyto_layout",
    style={"width": "100%", "height": "275px"},
    layout={"name": "preset"},
    elements=elements_init,
    stylesheet=[
        {
            "selector": "node",
            "style": {"content": "data(label)",
                      "width": "75px",
                      "height": "75px" },
        },
        {
            "selector": "edge",
            "style": {"width":20,"content": "data(weight)"},
        },
        {
            "selector": 'node[id ^= "a"], node[id ^= "b"], node[id ^= "c"], node[id ^= "d"], node[id ^= "e"], node[id ^= "f"], node[id ^= "g"], node[id ^= "h"], node[id ^= "i"]',
            "style": {
                "background-color": "orange"
            }
        },
        {
            "selector": 'node[id ^= "j"], node[id ^= "k"], node[id ^= "l"], node[id ^= "m"]',
            "style": {
                "background-color": "lightblue"
            }
        },
        {
            "selector": ".new-edge",
            "style": {
                "line-color": "red",
                "curve-style": "bezier",
                "control-point-step-size": 150,
                "loop-direction": "0deg",
                "loop-sweep": "90deg"
            }
        }
    ]
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1('MLB Pitcher Pitching Pattern Visualization System', style={'textAlign': 'center'})
    ], style={'width': '100%', 'height': '50px', 'margin': '2%'}),

    # Main layout
    html.Div([
        # Pitcher Name and Season
        html.Div([
            html.H3("Pitcher Name"),
            dcc.Dropdown(
                id='P-name',
                options=[
                    {'label': 'Shohei Ohtani', 'value': 'Ohtani'},
                    {'label': 'Yu Darvish', 'value': 'Darvish'},
                    {'label': 'Yusei Kikuchi', 'value': 'Kikuchi'},
                    {'label': 'Aaron Nola', 'value': 'Nola'},
                    {'label': 'Justin Verlander', 'value': 'Verlander,Justin'},
                    {'label': 'Carlos Rodon', 'value': 'Rodon'},
                    {'label': 'Gerrit Cole', 'value': 'Cole_Gerrit'},
                    {'label': 'Clayton Kershaw', 'value': 'Kershaw_Clayton'},
                    {'label': 'Chris Bassitt', 'value': 'Bassitt_Christopher'},
                    {'label': 'Spencer Strider', 'value': 'Strider_Spencer'},
                ],
                value='Ohtani',
            ),
            html.H3('Season'),
            dcc.Dropdown(
                id='Season',
                options=[
                    {'value': year, 'label': year}
                    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
                ],
                multi=True,
                value=[2020, 2021, 2022, 2023],
            )
        ], style={'width': '20%', 'margin': '3%', 'display': 'inline-block'}),

        # Stand Type and Number of Classes
        html.Div([
            html.H3('Stand Type'),
            dcc.RadioItems(
                id='Stype',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Left', 'value': 'L'},
                    {'label': 'Right', 'value': 'R'},
                ],
                value='All',
            ),
            html.H3('Number of Classes'),
            dcc.RadioItems(
                id='numc',
                options=[
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'},
                    {'label': '4', 'value': '4'},
                    {'label': '5', 'value': '5'},
                    {'label': '6', 'value': '6'},
                ],
                value='3',
            )
        ], style={'width': '20%', 'margin': '3%', 'display': 'inline-block'}),

        # Category Selection
        html.Div([
            html.H3('Category'),
            dcc.Checklist(
                id='Category',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Out', 'value': 'Out'},
                    {'label': 'StrikeOut', 'value': 'StrikeOut'},
                    {'label': 'BaseHit', 'value': 'BaseHit'},
                    {'label': 'Walk', 'value': 'Walk'},
                    {'label': 'HomeRun', 'value': 'HomeRun'},
                ],
            )
        ], style={'width': '20%', 'margin': '3%', 'display': 'inline-block'}),

        # Zone Selection and Controls
        html.Div([
            html.Div([
                html.H3("Zone Selection", style={"textAlign": "left", "display": "inline-block", "marginRight": "10px"}),
                html.Button('Start', id='Start-button', n_clicks=0, style={"display": "inline-block", 'margin': '1%'}),
                html.Button('Reset', id='reset-button', n_clicks=0, style={"display": "inline-block", 'margin': '1%'}),
                html.Button('Elbow', id='Elbow-button', n_clicks=0, style={"display": "inline-block", 'margin': '1%'}),
            ]),
            cyto_compo1,
            html.Div(id='selected-zones', style={'marginTop': '20px'}),
        ], style={'width': '20%', 'height': '375px', 'margin': '1%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexDirection': 'row'}),

    # Image Display
    html.Div(id='image-display', style={'width': '100%', 'textAlign': 'center', 'marginTop': '20px'}),

    # Store Component
    dcc.Store(id='selected-nodes', data=[])
])

@app.callback(
    [Output('selected-zones', 'children'),
     Output('image-display', 'children'),
     Output('selected-nodes', 'data'),
     Output('dash_cyto_layout', 'elements')],
    [
        Input('Start-button', 'n_clicks'),
        Input('reset-button', 'n_clicks'),
        Input('dash_cyto_layout', 'tapNodeData'),
    ],
    [
        State('P-name', 'value'),
        State('Season', 'value'),
        State('Stype', 'value'),
        State('numc', 'value'),
        State('Category', 'value'),
        State('selected-nodes', 'data'),
        State('dash_cyto_layout', 'elements')
    ]
)
def handle_buttons(start_clicks, reset_clicks, tapped_node, pitcher_name, year, stance_type, num_classes, Category, selected_nodes, elements):
    nodesize_unit = 50  # デフォルトのノードサイズを設定
    ctx = dash.callback_context

    selected_zones = 'Selected Zone : '
    image_element = []
    output_nodes = selected_nodes
    output_elements = elements

    if not ctx.triggered:
        return selected_zones, image_element, [], elements_init

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button' and reset_clicks > 0:
        if hasattr(update_elements, 'selected_nodes'):
            delattr(update_elements, 'selected_nodes')
        return selected_zones, image_element, [], elements_init

    if button_id == 'Start-button' and start_clicks > 0:
        for cat in Category:
            filtered_data = select_extraction_data(pitcher_name, year, stance_type, selected_nodes, cat)
            clustered_data = clustering_data(filtered_data, num_classes)
            kmedoids_fits = kmedoids_fit(filtered_data, clustered_data)

            create_NX(kmedoids_fits,
                      pitcher_name,
                      year,
                      stance_type,
                      selected_nodes,
                      num_classes,
                      cat,
                      nodesize_unit
                     )

        selected_zones = 'Selected Zone : ' + ', '.join(selected_nodes)
        image_element = update_NX(pitcher_name, year, stance_type, selected_nodes, num_classes, Category)
        output_nodes = selected_nodes
        output_elements = elements  # elementsは更新なし

    elif tapped_node is not None:
        output_elements, selected_zones, output_nodes = update_elements(
            tapped_node, elements=elements, selected_nodes=selected_nodes
        )
        image_element = []

    return selected_zones, image_element, output_nodes, output_elements

def update_elements(tapped_node, elements, selected_nodes):
    if tapped_node is None:
        return elements, 'Selected Zone : ' + ', '.join(selected_nodes), selected_nodes
    
    print(selected_nodes)

    new_elements = elements.copy()

    node_id = tapped_node['id']
    print("Clicked Node:", node_id)

    # ノードが選択された順番を保持するためのリスト
    if not hasattr(update_elements, 'selected_nodes'):
       update_elements.selected_nodes = []  # 初回呼び出し時にselected_nodesリストを作成
    
    update_elements.selected_nodes.append(node_id)
    print("Updated selected nodes after click:", update_elements.selected_nodes)

    # 最後の2つのノードからエッジを生成
    if len(update_elements.selected_nodes) > 1:
        source, target = update_elements.selected_nodes[-2], update_elements.selected_nodes[-1]
        new_edge = {
            "data": {"source": source, "target": target, "id": f"{source}-{target}"},
            "classes": "new-edge"
        }
        new_elements.append(new_edge)
        print(f"New edge created: {source} -> {target}")
    
    # ノードの選択状態を更新
    for node in new_elements:
        if 'data' in node and 'id' in node['data']:
            node['data']['selected'] = node['data']['id'] in update_elements.selected_nodes

    print(update_elements.selected_nodes)
    selected_zones = 'Selected Zone : ' + ', '.join(update_elements.selected_nodes)

    return new_elements, selected_zones, update_elements.selected_nodes

if __name__ == '__main__':
    app.run_server(debug=True)