import dash
import base64
import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from sklearn_extra.cluster import KMedoids
from dash import Dash, html, dcc, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
from math import sqrt
from itertools import product
from collections import Counter
import pickle

# ==============================================================================
# 1. Initialization and Global Settings
# ==============================================================================

# Initialize Dash App
app = dash.Dash(__name__)
server = app.server

# --- IMPORTANT: Please modify these paths to match your own environment. ---
image_directory = '/Users/tsujistencia/Desktop/Cash'
image_extension = '.png'
data_directory_base = '/Users/tsujistencia/Desktop/Player_Data' # A folder containing player CSV data files
cache_directory = '/Users/tsujistencia/Desktop/Cash/Cache_CSV'

# Create directories if they don't exist
if not os.path.exists(image_directory):
    os.makedirs(image_directory)
if not os.path.exists(data_directory_base):
    os.makedirs(data_directory_base)
    print(f"Warning: Data directory '{data_directory_base}' did not exist...")
if not os.path.exists(cache_directory): # ★★★ ディレクトリ作成処理を追加 ★★★
    os.makedirs(cache_directory)


# Pitch Categorization Maps
event_map00 = {
    '4-Seam Fastball':'Fastball', 'Sinker':'Fastball', 'Cutter':'Fastball',
    'Changeup':'Offspeed', 'Split-Finger':'Offspeed', 'Forkball':'Offspeed', 'Screwball':'Offspeed',
    'Curveball':'Curveball Group', 'Knuckle Curve':'Curveball Group', 'Slow Curve':'Curveball Group',
    'Slider':'Slider Group', 'Sweeper':'Slider Group', 'Slurve':'Slider Group',
    'Knuckleball':'Knuckle Group',
    'Eephus':'Others', 'Other':'Others', 'Intentional Ball':'Others', 'Pitch Out':'Others'
}
category_map = {
    'Fastball': 'F', 'Curveball Group': 'C', 'Offspeed': 'D',
    'Others': 'O', 'Slider Group': 'S'
}
result_event_map = {
    'stolen_base_2b': 'Out', 'catcher_interf': 'Walk', 'double_play': 'Out',
    'field_out': 'Out', 'fielders_choice_out': 'Out', 'force_out': 'Out',
    'grounded_into_double_play': 'Out', 'strikeout': 'StrikeOut',
    'strikeout_double_play': 'StrikeOut', 'double': 'BaseHit', 'hit_by_pitch': 'Walk',
    'single': 'BaseHit', 'triple': 'BaseHit', 'walk': 'Walk', 'home_run': 'HomeRun',
}

# Coordinate systems for distance calculation
node_position = {
    "a": (-1, 1), "b": (0, 1), "c": (1, 1), "d": (-1, 0), "e": (0, 0), "f": (1, 0),
    "g": (-1, -1), "h": (0, -1), "i": (1, -1), "j": (-2, 2), "k": (2, 2),
    "l": (-2, -2), "m": (2, -2), "n": (2, -2) # n is same as m, can be ignored
}
speed_coordinates = {"S": (-1, 0), "M": (0, 0), "H": (1, 0)}
# Pitch type coordinates will be generated dynamically based on data


# ==============================================================================
# 2. Data Processing and Distance Calculation Functions
# ==============================================================================

def weighted_edit_distance(s1, s2, cost_func, ins_del_cost=1.0):
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1): dp[i, 0] = i * ins_del_cost
    for j in range(n + 1): dp[0, j] = j * ins_del_cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = cost_func(s1[i-1], s2[j-1])
            dp[i, j] = min(dp[i-1, j] + ins_del_cost, dp[i, j-1] + ins_del_cost, dp[i-1, j-1] + sub_cost)
    return dp[m, n]

def get_trajectory_cost(char1, char2, node_pos=node_position):
    if char1 not in node_pos or char2 not in node_pos: return 1.0
    return sqrt((node_pos[char2][0] - node_pos[char1][0])**2 + (node_pos[char2][1] - node_pos[char1][1])**2)

def get_speed_cost(char1, char2, speed_pos=speed_coordinates):
    if char1 not in speed_pos or char2 not in speed_pos: return 1.0
    return abs(speed_pos[char1][0] - speed_pos[char2][0])

def get_pitch_type_cost(char1, char2, pitch_coords):
    if char1 == 'O' or char2 == 'O': return 0.0
    if char1 not in pitch_coords or char2 not in pitch_coords: return 1.0
    return abs(pitch_coords[char1][0] - pitch_coords[char2][0])

def normalize_matrix(matrix):
    min_val, max_val = np.min(matrix), np.max(matrix)
    if max_val - min_val > 0: return (matrix - min_val) / (max_val - min_val)
    return matrix

# ==============================================================================
# 3. Dash App Layout
# ==============================================================================

nodes_init_data = [{"data": {"id": k, "label": k.upper()}, "position": {"x": (v[0]+3)*100, "y": (-v[1]+3)*100}, "locked": True} for k, v in node_position.items() if k != 'n']
elements_init = nodes_init_data[:]

app.layout = html.Div([
    html.H1('MLB Pitcher‘s Pitching Patterns Visualization System', style={'textAlign': 'center', 'fontFamily': 'Helvetica'}),
    dcc.Store(id='selected-nodes-store', data=[]),
    dcc.Store(id='pitch-type-coords-store', data={}),
    html.Div([
        html.Div([
            html.H3("Pitcher & Data Selection"),
            dcc.Dropdown(id='P-name', options=[{'label': 'Shohei Ohtani', 'value': 'Ohtani'}, 
                                               {'label': 'Yu Darvish', 'value': 'Darvish'}, 
                                               {'label': 'Yusei Kikuchi', 'value': 'Kikuchi'}, 
                                               {'label': 'Gerrit Cole', 'value': 'Cole_Gerrit'},
                                               {'label':'Max Scherzer', 'value':'Scherzer'}, 
                                               {'label':'Kyle Gibson', 'value':'Gibson'}, 
                                               {'label': 'Clayton Kershaw', 'value': 'Kershaw_Clayton'}, 
                                               {'label': 'Justin Verlander', 'value': 'Verlander_Justin'}, 
                                               {'label': 'Carlos Rodon', 'value': 'Rodon'}, 
                                               {'label': 'Chris Bassitt', 'value': 'Bassitt_Christopher'}, 
                                               {'label': 'Aaron Nola', 'value': 'Nola'}], 
                                               value='Ohtani'),
            html.H3('Season'),
            dcc.Dropdown(id='Season', options=[{'value': y, 'label': str(y)} for y in range(2015, 2025)], multi=True, value=[2023]),
            html.H3('Batter Stance'),
            dcc.RadioItems(id='Stype', options=[{'label': k, 'value': v} for k, v in {'All': 'All', 'vs Left': 'L', 'vs Right': 'R'}.items()], value='All'),
            html.H3('Number of Clusters'),
            dcc.RadioItems(id='numc', options=[{'label': str(i), 'value': str(i)} for i in range(2, 7)], value='3', inline=True),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3('Filter by Result'),
            dcc.Checklist(id='Category', options=[{'label': cat, 'value': cat} for cat in ['All', 'Out', 'StrikeOut', 'BaseHit', 'Walk', 'HomeRun']], value=['All'], inline=True),
            html.Hr(),
            html.H3("Distance Weights"),
            html.Label("Trajectory (Location) Weight"),
            dcc.Slider(id='w-traj-slider', min=0, max=1, step=0.5, value=1, marks={i/2: str(i/2) for i in range(2)}),
            html.Label("Speed (S/M/H) Weight"),
            dcc.Slider(id='w-speed-slider', min=0, max=1, step=0.5, value=1, marks={i/2: str(i/2) for i in range(2)}),
            html.Label("Pitch Type Weight"),
            dcc.Slider(id='w-pitch-slider', min=0, max=1, step=0.5, value=1, marks={i/2: str(i/2) for i in range(2)}),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Zone Selection"),
            html.Div(id='selected-zones-text', style={'minHeight': '20px', 'fontWeight': 'bold'}),
            cyto.Cytoscape(id="dash-cytoscape", style={"width": "100%", "height": "250px"}, layout={"name": "preset"}, elements=elements_init,
                stylesheet=[
                    {"selector": "node", "style": {"content": "data(label)", "width": "25px", "height": "25px", 'font-size': '10px'}},
                    {'selector': 'node[id *= "j"], node[id *= "k"], node[id *= "l"], node[id *= "m"]', "style": {"background-color": "lightblue"}},
                    {'selector': 'node[id ^= "a"], node[id ^= "b"], node[id ^= "c"], node[id ^= "d"], node[id ^= "e"], node[id ^= "f"], node[id ^= "g"], node[id ^= "h"], node[id ^= "i"]', "style": {"background-color": "orange"}},
                    {"selector": ".new-edge", "style": {"line-color": "red", "target-arrow-color": "red", "target-arrow-shape": "triangle", "curve-style": "bezier", "width": 5, "control-point-step-size": 70 }},
                ]
            ),
            html.Button('Start Analysis', id='start-button', n_clicks=0, style={'marginRight': '10px', 'marginTop': '10px'}),
            html.Button('Reset Selection', id='reset-button', n_clicks=0, style={'marginTop': '10px'}),
        ], style={'width': '30%', 'padding': '5px', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Graph Display"),
            html.Label("Node Size Scale"),
            dcc.Slider(id='nodesize-slider', min=10, max=100, step=10, value=40, marks={i: str(i) for i in range(10, 101, 10)}),
            html.Label("Edge Size Scale"),
            dcc.Slider(id='edgesize-slider', min=10, max=100, step=10, value=40, marks={i: str(i) for i in range(10, 101, 10)}),
            html.Label("Minimum Edge Frequency"),
            dcc.Slider(id='min-edge-weight-slider', min=1, max=55, step=5, value=20, marks={i: str(i) for i in range(0, 51, 5)}),
        ], style={'width': '20%', 'padding': '5px', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Hr(),
    dcc.Loading(id="loading-spinner", children=[html.Div(id='image-display')], type="circle"),
])

# ==============================================================================
# 4. Core Logic and Helper Functions
# ==============================================================================

# --- Data Processing Functions ---
def select_and_preprocess_data(name, season_list, Stype):
    path = os.path.join(data_directory_base, f'{name}.csv')
    if not os.path.exists(path): return pd.DataFrame(), {}
    try: data = pd.read_csv(path)
    except Exception as e: return pd.DataFrame(), {}
    data['game_date'] = pd.to_datetime(data['game_date'])
    data['year'] = data['game_date'].dt.year
    if not isinstance(season_list, list) or not season_list: season_list = [2023]
    data = data[data['year'].isin(season_list)]
    if Stype != 'All': data = data[data['stand'] == Stype]
    if data.empty: return pd.DataFrame(), {}
    data['Pitch_Category_T'] = data['pitch_name'].map(event_map00).fillna(data['pitch_name'])
    data['PitchCategory'] = data['Pitch_Category_T'].map(category_map).fillna('U')
    all_speeds = data['effective_speed'].dropna()
    low, high = np.percentile(all_speeds, [33, 66]) if len(all_speeds) >= 3 else (80, 90)
    def categorize_speed(speed):
        if pd.isna(speed): return 'M'
        if speed <= low: return 'S'
        elif speed <= high: return 'M'
        else: return 'H'
    data['CategorizedSpeed'] = data['effective_speed'].apply(categorize_speed)
    speed_ave = data.groupby('PitchCategory')['effective_speed'].mean()
    min_speed, max_speed = speed_ave.min(), speed_ave.max()
    pitch_type_coords = {p: (((s - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.0), 0) for p, s in speed_ave.items()}
    zone_map = {1:'c', 2:'b', 3:'a', 4:'f', 5:'e', 6:'d', 7:'i', 8:'h', 9:'g', 11:'k', 12:'j', 13:'m', 14:'l'}
    data['zone_char'] = data['zone'].map(zone_map).fillna('')
    data.sort_values(by=['game_date', 'at_bat_number', 'pitch_number'], inplace=True)
    rows = []
    for _, g in data.groupby(['game_date', 'at_bat_number']):
        if len(g) > 1:
            rows.append({'trajectory': ''.join(g['zone_char']), 'SpeedSequence': ''.join(g['CategorizedSpeed']), 'PitchSequence': ''.join(g['PitchCategory']), 'Result': g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None})
    if not rows: return pd.DataFrame(), {}
    processed_df = pd.DataFrame(rows)
    processed_df['Category'] = processed_df['Result'].map(result_event_map).fillna('Other')
    return processed_df, pitch_type_coords

def perform_clustering(df, num_classes_str, w_traj, w_speed, w_pitch, pitch_coords):
    trajectories, n, num_classes = df.to_dict('records'), len(df), int(num_classes_str)
    if n < num_classes: num_classes = n
    if num_classes == 0: return np.array([])
    dist_traj, dist_speed, dist_pitch = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist_traj[i, j] = dist_traj[j, i] = weighted_edit_distance(trajectories[i]['trajectory'], trajectories[j]['trajectory'], get_trajectory_cost, ins_del_cost=2.83)
            dist_speed[i, j] = dist_speed[j, i] = weighted_edit_distance(trajectories[i]['SpeedSequence'], trajectories[j]['SpeedSequence'], get_speed_cost, ins_del_cost=1.0)
            cost_func = lambda c1, c2: get_pitch_type_cost(c1, c2, pitch_coords)
            dist_pitch[i, j] = dist_pitch[j, i] = weighted_edit_distance(trajectories[i]['PitchSequence'], trajectories[j]['PitchSequence'], cost_func, ins_del_cost=0.5)
    final_dist = (w_traj * normalize_matrix(dist_traj) + w_speed * normalize_matrix(dist_speed) + w_pitch * normalize_matrix(dist_pitch))
    kmedoids = KMedoids(n_clusters=num_classes, random_state=0, metric='precomputed', init='k-medoids++')
    try:
        kmedoids.fit(final_dist)
        return kmedoids.labels_
    except Exception as e:
        print(f"Error during KMedoids fitting: {e}")
        return np.array([])

# --- Graph Generation Functions ---
def class_choice(clustered_data, class_label): return clustered_data[clustered_data['class'] == class_label]['trajectory']
def nodeCount0(trajectories): return Counter(char for traj in trajectories for char in traj)
def edgeCount(trajectories): return Counter(edge for traj in trajectories if len(traj) > 1 for edge in zip(traj[:-1], traj[1:]))
def edge_merge(sorted_edges_with_counts):
    merged = {}
    for edge, count in sorted_edges_with_counts:
        canonical_edge = tuple(sorted(edge))
        merged[canonical_edge] = merged.get(canonical_edge, 0) + count
    return merged

def create_NX(clustered_data, pitcher_name, year_str, stance_type, seq_str, num_classes, category, nodesize_unit, min_edge_weight):
    allowed_nodes, s_zone, b_zone = list('abcdefghijklm'), list('abcdefghi'), list('jklm')
    for i in range(int(num_classes)):
        test_data = class_choice(clustered_data, i)
        if test_data.empty: continue
        
        node00, edge01 = nodeCount0(test_data), edgeCount(test_data)
        dic02 = sorted(edge01.items(), key=lambda x: x[1], reverse=True)
        edge_data00 = edge_merge(dic02)
        
        G = nx.Graph()
        G.add_nodes_from([(n, {"color": "darkorange" if n in s_zone else "skyblue"}) for n in allowed_nodes])
        nodesize = [nodesize_unit * node00.get(node_id, 0) if node00.get(node_id, 0) > 0 else nodesize_unit/2 for node_id in G.nodes]
        
        for k, v in edge_data00.items():
            if v < min_edge_weight or k[0] == k[1]:
                continue
            if (k[0] not in allowed_nodes) or (k[1] not in allowed_nodes):
                continue
            
            if (k[0] in b_zone and k[1] in b_zone): color = 'b'
            elif (k[0] in s_zone and k[1] in s_zone): color = 'r'
            else: color = 'g'
            G.add_edge(k[0], k[1], weight=v, color=color)
            
        pos = {'a':[-1.2,1.5], 'b':[0,1.9], 'c':[1.2,1.5], 'd':[-1.7,0], 'e':[0.1,0.3], 'f':[1.7,0], 'g':[-1.2,-1.5], 'h':[0,-1.9], 'i':[1.2,-1.5], 'j':[-3,2], 'k':[3,2], 'l':[-3,-2], 'm':[3,-2]}
        
        filename = f'{pitcher_name}_{year_str}_{stance_type}_{category}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
        update_Path = os.path.join(image_directory, filename)
        
        if os.path.exists(update_Path):
            print(f"File already exists, skipping: {filename}")
            continue  # 存在すれば描画をスキップして次のループへ

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_size=nodesize, node_color=[n["color"] for n in G.nodes.values()], edge_color=[G[u][v].get('color','gray') for u,v in G.edges], width=[w*0.5 for w in [G[u][v].get('weight',1) for u,v in G.edges]], font_size = 20)
        plt.title(f"Cluster {i+1} for {category} ({len(test_data)} sequences)")
        plt.savefig(update_Path, bbox_inches='tight')
        plt.close('all')
        print('生成終了')

# --- New Helper Functions for Combined Callback ---
def update_elements(tapped_node, selected_nodes):
    node_id = tapped_node['id']
    new_selected_nodes = selected_nodes + [node_id]
    current_nodes = [{"data": {"id": k, "label": k.upper()}, "position": {"x": (v[0]+3)*100, "y": (-v[1]+3)*100}, "locked": True, "classes": ''} for k, v in node_position.items() if k != 'n']
    current_edges = []
    if len(new_selected_nodes) > 1:
        for i in range(len(new_selected_nodes) - 1):
            source, target = new_selected_nodes[i], new_selected_nodes[i+1]
            current_edges.append({"data": {"source": source, "target": target}, "classes": "new-edge"})
    new_elements = current_nodes + current_edges
    text_output = 'Selected Zone: ' + ' -> '.join(new_selected_nodes)
    return new_elements, text_output, new_selected_nodes

def update_image_display(pitcher_name, seasons, stance_type, selected_zones, num_classes, categories, min_edge_weight):
    year_str = "_".join(map(str, sorted(seasons)))
    seq_str = "".join(selected_zones) if selected_zones else "any"
    
    categories_to_process = categories
    if 'All' in categories:
        categories_to_process = ['All']
    else:
        categories_to_process = categories

    category_blocks = []
    for cat in categories_to_process:
        cat_images = []
        for i in range(int(num_classes)):
            # ★★★ このファイル名形式に create_NX を合わせました ★★★
            filename = f'{pitcher_name}_{year_str}_{stance_type}_{cat}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
            image_path = os.path.join(image_directory, filename)
            if os.path.exists(image_path):
                try:
                    encoded_img = base64.b64encode(open(image_path, 'rb').read()).decode()
                    cat_images.append(html.Img(src=f'data:image/png;base64,{encoded_img}', style={'height': '300px', 'margin': '5px'}))
                except Exception as e:
                    print(f"Error encoding image {image_path}: {e}")
        if cat_images:
            block = html.Div([html.H3(f"Results for Category: {cat}", style={'textAlign': 'center'}), html.Div(cat_images, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})], style={'width': '49%', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'margin': '0.5%'})
            category_blocks.append(block)

    if not category_blocks:
        return html.Div("No patterns found for the selected filters.")

    final_layout = []
    for i in range(0, len(category_blocks), 2):
        row_children = [category_blocks[i]]
        if (i + 1) < len(category_blocks):
            row_children.append(category_blocks[i + 1])
        row = html.Div(row_children, style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
        final_layout.append(row)
    return html.Div(final_layout)


# ==============================================================================
# 5. Dash Callbacks (修正版：自己ループと選択のバグを修正)
# ==============================================================================

# Callback 1: Cytoscapeのインタラクティブな操作を正しく処理する
@app.callback(
    Output('selected-nodes-store', 'data'),
    Output('selected-zones-text', 'children'),
    Output('dash-cytoscape', 'elements'),
    Input('dash-cytoscape', 'tapNodeData'),
    Input('reset-button', 'n_clicks'),
    State('selected-nodes-store', 'data'),
    prevent_initial_call=True
)
def handle_cyto_tap(tapped_node, reset_clicks, selected_nodes):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # リセットボタンが押されたら初期状態に戻す
    if triggered_id == 'reset-button':
        return [], 'Selected Zone: ', elements_init

    # ノードがタップされた場合の処理
    if triggered_id == 'dash-cytoscape' and tapped_node:
        node_id = tapped_node['id']
        
        # 同じノードの連続選択を許可し、シーケンスに追加
        new_selected_nodes = selected_nodes + [node_id]

        # 毎回ゼロから要素（ノードとエッジ）を再生成する
        current_nodes = [
            {"data": {"id": k, "label": k.upper()}, "position": {"x": (v[0]+3)*100, "y": (-v[1]+3)*100}, "locked": True, "classes": ''}
            for k, v in node_position.items() if k != 'n'
        ]
        
        current_edges = []
        if len(new_selected_nodes) > 1:
            for i in range(len(new_selected_nodes) - 1):
                source, target = new_selected_nodes[i], new_selected_nodes[i+1]
                current_edges.append({"data": {"source": source, "target": target}, "classes": "new-edge"})

        new_elements = current_nodes + current_edges
        text_output = 'Selected Zone: ' + ' -> '.join(new_selected_nodes)
        
        # 【バグ修正点】計算した新しい要素と選択リストを正しく返す
        return new_selected_nodes, text_output, new_elements

    # ノードタップ以外のトリガー（起動時など）は何もしない
    return no_update, no_update, no_update


# Callback 2: 「Start Analysis」ボタンによる重い処理（変更なし）
@app.callback(
    Output('image-display', 'children'),
    Output('pitch-type-coords-store', 'data'),
    Input('start-button', 'n_clicks'),
    State('P-name', 'value'), State('Season', 'value'), State('Stype', 'value'),
    State('Category', 'value'), State('numc', 'value'),
    State('w-traj-slider', 'value'), State('w-speed-slider', 'value'), State('w-pitch-slider', 'value'),
    State('nodesize-slider', 'value'), State('min-edge-weight-slider', 'value'),
    State('selected-nodes-store', 'data'),
    State('pitch-type-coords-store', 'data'),
    prevent_initial_call=True
)

def run_analysis(n_clicks, p_name, seasons, s_type, categories, num_c,
                 w_t, w_s, w_p, node_size, min_edge,
                 selected_nodes, pitch_coords):

    # 1. データの前処理（ここはキャッシュしない、または前回同様のキャッシュでもOK）
    full_df, new_pitch_coords = select_and_preprocess_data(p_name, seasons, s_type)
    if full_df.empty:
        return html.Div(f"No data found for {p_name}."), no_update

    # 2. ゾーン指定によるフィルタリング
    prefix_str = "".join(selected_nodes)
    if prefix_str:
        filtered_by_prefix_df = full_df[full_df['trajectory'].str.startswith(prefix_str)].copy()
        if filtered_by_prefix_df.empty:
            return html.Div(f"No data matches the selected prefix '{prefix_str}'."), no_update
    else:
        filtered_by_prefix_df = full_df.copy()

    # 3. 解析カテゴリの決定
    if 'All' in categories:
        categories_to_process = ['All']
    else:
        categories_to_process = categories

    year_str = "_".join(map(str, sorted(seasons)))
    seq_str = "".join(selected_nodes) if selected_nodes else "any"

    # 4. カテゴリごとにクラスタリングと画像生成
    for cat in categories_to_process:
        # ▼▼▼【クラスタリング結果のキャッシュ機能】ここから ▼▼▼
        # パラメータを全て含んだユニークなファイル名を生成
        cache_params = f"{p_name}_{year_str}_{s_type}_{cat}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
        cache_filename = f"clustered_{cache_params}.pkl"
        cache_filepath = os.path.join(cache_directory, cache_filename)

        if os.path.exists(cache_filepath):
            print(f"--- Loading CLUSTERED data from cache: {cache_filename} ---")
            try:
                with open(cache_filepath, 'rb') as f:
                    # ラベル付け済みのcat_dfを直接読み込む
                    cat_df_with_labels = pickle.load(f)
            except Exception as e:
                print(f"Cache load failed: {e}. Re-clustering...")
                # 失敗した場合はキャッシュを無視して再計算
                cat_df_with_labels = None
        else:
            cat_df_with_labels = None
            print(f"--- Clustered cache not found. Performing clustering for: {cat} ---")
        
        if cat_df_with_labels is None: # キャッシュが存在しないか、読み込みに失敗した場合
            # クラスタリング前のデータフレームを準備
            if cat == 'All':
                cat_df = filtered_by_prefix_df.reset_index(drop=True)
            else:
                cat_df = filtered_by_prefix_df[filtered_by_prefix_df['Category'] == cat].reset_index(drop=True)
            
            if cat_df.empty:
                continue

            # 最も重い処理を実行
            labels = perform_clustering(cat_df, num_c, w_t, w_s, w_p, new_pitch_coords)
            if labels.size == 0:
                continue
            
            cat_df['class'] = labels
            cat_df_with_labels = cat_df # ラベルを付けたものを保存対象に
            
            # 結果をキャッシュとして保存
            print(f"--- Saving CLUSTERED data to cache: {cache_filename} ---")
            with open(cache_filepath, 'wb') as f:
                pickle.dump(cat_df_with_labels, f)

        # ▲▲▲【クラスタリング結果のキャッシュ機能】ここまで ▲▲▲

        if cat_df_with_labels.empty:
            continue
            
        # ネットワーク図を生成（入力はラベル付け済みのDF）
        create_NX(cat_df_with_labels, p_name, year_str, s_type, seq_str, num_c, cat, node_size, min_edge)

    # 5. 表示用HTML要素の生成
    image_elements = update_image_display(p_name, seasons, s_type, selected_nodes, num_c, categories, min_edge)

    return image_elements, new_pitch_coords
# ==============================================================================
# 6. Run the Application
# ==============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)