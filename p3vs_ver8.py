import base64
import io
import json
import os
import logging
import pickle
from hashlib import md5
from math import sqrt
from collections import Counter
from pathlib import Path

import dash
import dash_cytoscape as cyto
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, no_update
from dash import dash_table
from dash.dependencies import Input, Output, State, ALL
from sklearn_extra.cluster import KMedoids
from matplotlib.colors import to_rgba
from flask import Response

# ======================================================================
# 0) メモリ上のPNGキャッシュ（保存しない）
# ======================================================================
NETWORK_IMAGE_CACHE = {}

# ======================================================================
# 1) Initialization and Global Settings
# ======================================================================

app = dash.Dash(__name__)
server = app.server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORTANT: paths ---
HOME = Path.home()
image_directory = Path(os.path.join(HOME, 'Desktop', 'Cash'))
data_directory_base = Path(os.path.join(HOME, 'Desktop', 'Player_Data'))
cache_directory = image_directory / 'Cache_CSV'

def ensure_dirs(paths):
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            import os as _os
            _os.makedirs(str(p), exist_ok=True)

ensure_dirs([image_directory, data_directory_base, cache_directory])

# Pitch Categorization Maps
event_map00 = {
    '4-Seam Fastball':'Fastball', 'Sinker':'Fastball', 'Cutter':'Fastball',
    'Changeup':'Offspeed', 'Split-Finger':'Offspeed', 'Forkball':'Offspeed', 'Screwball':'Offspeed',
    'Curveball':'Curveball Group', 'Knuckle Curve':'Curveball Group', 'Slow Curve':'Curveball Group',
    'Slider':'Slider Group', 'Sweeper':'Slider Group', 'Slurve':'Slider Group',
    'Knuckleball':'Knuckle Group',
    'Eephus':'Others', 'Other':'Others', 'Intentional Ball':'Others', 'Pitch Out':'Others'
}
category_map = {'Fastball': 'F', 'Curveball Group': 'C', 'Offspeed': 'D', 'Others': 'O', 'Slider Group': 'S'}
result_event_map = {
    'stolen_base_2b': 'Out', 'catcher_interf': 'Walk', 'double_play': 'Out',
    'field_out': 'Out', 'fielders_choice_out': 'Out', 'force_out': 'Out',
    'grounded_into_double_play': 'Out', 'strikeout': 'StrikeOut',
    'strikeout_double_play': 'StrikeOut', 'double': 'BaseHit', 'hit_by_pitch': 'Walk',
    'single': 'BaseHit', 'triple': 'BaseHit', 'walk': 'Walk', 'home_run': 'HomeRun',
}

# Coordinates
node_position = {
    "a": (-1, 1), "b": (0, 1), "c": (1, 1), "d": (-1, 0), "e": (0, 0), "f": (1, 0),
    "g": (-1, -1), "h": (0, -1), "i": (1, -1), "j": (-2, 2), "k": (2, 2),
    "l": (-2, -2), "m": (2, -2)
}
speed_coordinates = {"S": (-1, 0), "M": (0, 0), "H": (1, 0)}

# ======================================================================
# 2) Distance Utilities
# ======================================================================

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

def get_trajectory_cost(c1, c2, node_pos=node_position):
    if c1 not in node_pos or c2 not in node_pos: return 1.0
    return sqrt((node_pos[c2][0] - node_pos[c1][0])**2 + (node_pos[c2][1] - node_pos[c1][1])**2)

def get_speed_cost(c1, c2, speed_pos=speed_coordinates):
    if c1 not in speed_pos or c2 not in speed_pos: return 1.0
    return abs(speed_pos[c1][0] - speed_pos[c2][0])

def get_pitch_type_cost(c1, c2, pitch_coords):
    if c1 == 'O' or c2 == 'O': return 0.0
    if c1 not in pitch_coords or c2 not in pitch_coords: return 1.0
    return abs(pitch_coords[c1][0] - pitch_coords[c2][0])

def normalize_matrix(matrix):
    min_val, max_val = np.min(matrix), np.max(matrix)
    if max_val - min_val > 0: return (matrix - min_val) / (max_val - min_val)
    return matrix

# ======================================================================
# 3) Layout
# ======================================================================

nodes_init_data = [{"data": {"id": k, "label": k.upper()},
                    "position": {"x": (v[0]+3)*100, "y": (-v[1]+3)*100},
                    "locked": True} for k, v in node_position.items()]
elements_init = nodes_init_data[:]

app.layout = html.Div([
    html.H1('MLB Pitcher‘s Pitching Patterns Visualization System', style={'textAlign': 'center', 'fontFamily': 'Helvetica'}),
    dcc.Store(id='selected-nodes-store', data=[]),
    dcc.Store(id='pitch-type-coords-store', data={}),
    dcc.Store(id='cluster-store', data={}),   # {'Cat_i':[{'trajectory','PitchSequence','SpeedSequence'}]}
    dcc.Store(id='network-url'),              # 自動で開くURL
    dcc.Store(id='network-launch'),           # JS起動のダミー
    html.Div([
        html.Div([
            html.H3("Pitcher & Data Selection"),
            dcc.Dropdown(id='P-name', options=[
                {'label': 'Shohei Ohtani', 'value': 'Ohtani'},
                {'label': 'Yu Darvish', 'value': 'Darvish'},
                {'label': 'Yusei Kikuchi', 'value': 'Kikuchi'},
                {'label':'Blake Snell', 'value':'Snell'},
                {'label': 'Gerrit Cole', 'value': 'Cole_Gerrit'},
                {'label': 'Max Scherzer', 'value': 'Scherzer'},
                {'label': 'Kyle Gibson', 'value': 'Gibson'},
                {'label': 'Clayton Kershaw', 'value': 'Kershaw'},
                {'label': 'Justin Verlander', 'value': 'Verlander_Justin'},
                {'label': 'Carlos Rodon', 'value': 'Rodon'},
                {'label': 'Chris Bassitt', 'value': 'Bassitt_Christopher'},
                {'label': 'Aaron Nola', 'value': 'Nola'}
            ], value='Ohtani'),
            html.H3('Season'),
            dcc.Dropdown(id='Season',
                         options=[{'value': y, 'label': str(y)} for y in range(2015, 2026)],
                         multi=True, value=[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]),
            html.H3('Batter Stance'),
            dcc.RadioItems(id='Stype',
                           options=[{'label': k, 'value': v} for k, v in {'All': 'All', 'Left': 'L', 'Right': 'R'}.items()],
                           value='All'),
            html.H3('Number of Clusters'),
            dcc.RadioItems(id='numc', options=[{'label': str(i), 'value': str(i)} for i in range(2, 7)],
                           value='4', inline=True),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3('Filter by Result'),
            dcc.Checklist(id='Category',
                          options=[{'label': cat, 'value': cat} for cat in ['All', 'Out', 'StrikeOut', 'BaseHit', 'Walk', 'HomeRun']],
                          value=['Out', 'StrikeOut'], inline=True),
            html.Hr(),
            html.H3("Distance Weights"),
            html.Label("Trajectory (Location) Weight"),
            dcc.Slider(id='w-traj-slider', min=0, max=1, step=0.5, value=1, marks={0:'0',0.5:'0.5',1:'1'}),
            html.Label("Speed (S/M/H) Weight"),
            dcc.Slider(id='w-speed-slider', min=0, max=1, step=0.5, value=1, marks={0:'0',0.5:'0.5',1:'1'}),
            html.Label("Pitch Type Weight"),
            dcc.Slider(id='w-pitch-slider', min=0, max=1, step=0.5, value=1, marks={0:'0',0.5:'0.5',1:'1'}),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3("Zone Selection"),
            html.Div(id='selected-zones-text', style={'minHeight': '20px', 'fontWeight': 'bold'}),
            cyto.Cytoscape(
                id="dash-cytoscape",
                style={"width": "100%", "height": "250px"},
                layout={"name": "preset"},
                elements=elements_init,
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

    # クラスタごとのテーブル＋クリックで別タブ表示
    html.Div(id='cluster-table', style={'marginTop': '20px'}),
    html.Div(id='seq-preview', style={'marginTop': '16px'})
])

# ======================================================================
# 4) Data Processing and Clustering
# ======================================================================

def select_and_preprocess_data(name, season_list, Stype):
    path = data_directory_base / f'{name}.csv'
    if not path.exists():
        logger.info('Data file not found: %s', path)
        return pd.DataFrame(), {}
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        logger.exception('Failed reading CSV %s: %s', path, exc)
        return pd.DataFrame(), {}
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
            rows.append({
                'trajectory': ''.join(g['zone_char']),
                'SpeedSequence': ''.join(g['CategorizedSpeed']),
                'PitchSequence': ''.join(g['PitchCategory']),
                'Result': g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None
            })
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
        logger.exception("KMedoids fitting error: %s", e)
        return np.array([])

# --- helpers for cluster images ---
def class_choice(clustered_data, class_label): return clustered_data[clustered_data['class'] == class_label]['trajectory']
def nodeCount0(trajectories): return Counter(char for traj in trajectories for char in traj)
def edgeCount(trajectories): return Counter(edge for traj in trajectories if len(traj) > 1 for edge in zip(traj[:-1], traj[1:]))

def edge_merge(sorted_edges_with_counts):
    merged = {}
    for edge, count in sorted_edges_with_counts:
        canonical = tuple(sorted(edge))
        merged[canonical] = merged.get(canonical, 0) + count
    return merged

# ======================================================================
# 4.5) Cluster Overview Network (赤は前面)
# ======================================================================
def create_NX(clustered_data, pitcher_name, year_str, stance_type, seq_str,
              num_classes, category, nodesize_unit, min_edge_weight):
    allowed_nodes = list('abcdefghijklm')
    s_zone = list('abcdefghi')   # strike zone
    b_zone = list('jklm')        # ball zone

    for i in range(int(num_classes)):
        test_data = class_choice(clustered_data, i)
        if test_data.empty:
            continue

        node00 = nodeCount0(test_data)
        edge01 = edgeCount(test_data)
        dic02 = sorted(edge01.items(), key=lambda x: x[1], reverse=True)
        edge_data00 = edge_merge(dic02)

        G = nx.Graph()
        G.add_nodes_from(
            (n, {"color": "darkorange" if n in s_zone else "skyblue"})
            for n in allowed_nodes
        )

        nodesize = [
            nodesize_unit * node00.get(n, 0) if node00.get(n, 0) > 0 else nodesize_unit / 2
            for n in G.nodes
        ]

        for (u, v), cnt in edge_data00.items():
            if cnt < min_edge_weight or u == v: continue
            if (u not in allowed_nodes) or (v not in allowed_nodes): continue

            if (u in b_zone) and (v in b_zone):
                base_color = '#1f6fe5'     # blue
            elif (u in s_zone) and (v in s_zone):
                base_color = '#d32f2f'     # red
            else:
                base_color = '#12a36e'     # green
            G.add_edge(u, v, weight=cnt, base_color=base_color)

        pos = {
            'a':[-1.2,1.5], 'b':[0,1.9], 'c':[1.2,1.5],
            'd':[-1.7,0],   'e':[0.1,0.3], 'f':[1.7,0],
            'g':[-1.2,-1.5],'h':[0,-1.9],  'i':[1.2,-1.5],
            'j':[-3,2],     'k':[3,2],     'l':[-3,-2], 'm':[3,-2]
        }

        max_w = max((data.get('weight', 1) for _, _, data in G.edges(data=True)), default=1)

        # 赤系とその他で分けて、描画順を制御
        red_edges, other_edges = [], []
        for u, v, data in G.edges(data=True):
            base_color = data.get('base_color', '#999999').lower()
            if base_color in ('red', '#d32f2f', '#c62828', '#e53935'):
                red_edges.append((u, v, data))
            else:
                other_edges.append((u, v, data))
        edge_list_ordered = other_edges + red_edges

        edge_colors_rgba, edge_widths = [], []
        for u, v, data in edge_list_ordered:
            w = data.get('weight', 1)
            base_color = data.get('base_color', '#999999')
            alpha_base = 0.25 + 0.75 * (w / max_w) if max_w > 0 else 1.0
            is_red = base_color in ('red', '#d32f2f', '#c62828', '#e53935')
            alpha = 1.0 if is_red else alpha_base
            edge_colors_rgba.append(to_rgba(base_color, alpha))
            edge_widths.append(max(1.5, 0.5 * w))

        filename = f'{pitcher_name}_{year_str}_{stance_type}_{category}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
        update_Path = image_directory / filename
        if update_Path.exists():
            logger.info('File already exists, skipping: %s', filename)
            continue

        plt.figure(figsize=(8, 8))
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=nodesize,
            node_color=[n_attr["color"] for _, n_attr in G.nodes(data=True)],
            edgelist=[(u, v) for u, v, _ in edge_list_ordered],
            edge_color=edge_colors_rgba,
            width=edge_widths,
            font_size=20
        )
        plt.title(f"Cluster {i+1} for {category} ({len(test_data)} sequences)")
        try:
            plt.savefig(update_Path, bbox_inches='tight')
        except Exception as exc:
            logger.exception('Failed to save image %s: %s', update_Path, exc)
        finally:
            plt.close('all')

# ======================================================================
# 5) Helper UIs
# ======================================================================

def update_image_display(pitcher_name, seasons, stance_type, selected_zones, num_classes, categories, min_edge_weight):
    year_str = "_".join(map(str, sorted(seasons)))
    seq_str = "".join(selected_zones) if selected_zones else "any"
    categories_to_process = ['All'] if 'All' in categories else categories

    category_blocks = []
    for cat in categories_to_process:
        cat_images = []
        for i in range(int(num_classes)):
            filename = f'{pitcher_name}_{year_str}_{stance_type}_{cat}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
            image_path = image_directory / filename
            if image_path.exists():
                try:
                    with image_path.open('rb') as fh:
                        encoded_img = base64.b64encode(fh.read()).decode()
                    cat_images.append(html.Img(src=f'data:image/png;base64,{encoded_img}', style={'height': '300px', 'margin': '5px'}))
                except Exception as exc:
                    logger.exception('Error encoding image %s: %s', image_path, exc)
        cluster_buttons = []
        for i in range(int(num_classes)):
            btn_id = {'type': 'cluster-btn', 'category': cat, 'cluster': i}
            cluster_buttons.append(html.Button(
                f'View sequences: {cat} - Cluster {i+1}',
                id=btn_id, n_clicks=0, style={'margin':'6px'}
            ))
        if cat_images or cluster_buttons:
            block = html.Div([
                html.H3(f"Results for Category: {cat}", style={'textAlign': 'center'}),
                html.Div(cat_images, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
                html.Div(cluster_buttons, style={'textAlign':'center'})
            ], style={'width': '49%', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'margin': '0.5%'})
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

# 個別のシーケンス図（PNG保存ありの簡易版／使わなくてもOK）
def draw_single_sequence_image(traj_str):
    if not traj_str or len(traj_str) < 2:
        return html.Div("Select one trajectory row."), None
    key = md5(traj_str.encode('utf-8')).hexdigest()[:12]
    fname = f"seq_{key}.png"
    fpath = image_directory / fname
    if not fpath.exists():
        pos = {k: ((v[0]+3)*1.0, (v[1]+3)*1.0) for k, v in node_position.items()}
        plt.figure(figsize=(6, 6))
        for nid, (x, y) in pos.items():
            color = 'orange' if nid in list('abcdefghi') else 'lightblue'
            plt.scatter(x, y, s=300, c=color, edgecolors='black')
            plt.text(x, y, nid.upper(), ha='center', va='center', fontsize=10, color='black')
        for i in range(len(traj_str)-1):
            s, t = traj_str[i], traj_str[i+1]
            if s not in pos or t not in pos: continue
            x1, y1 = pos[s]; x2, y2 = pos[t]
            plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
            xm, ym = (x1+x2)/2, (y1+y2)/2
            plt.text(xm, ym, str(i+1), fontsize=9)
        plt.axis('off')
        plt.tight_layout()
        try:
            plt.savefig(fpath, bbox_inches='tight', dpi=150)
        finally:
            plt.close('all')
    try:
        with fpath.open('rb') as fh:
            encoded = base64.b64encode(fh.read()).decode()
    except Exception:
        logger.exception('Failed to read generated sequence image: %s', fpath)
        return html.Div('Could not open generated image.'), None
    return html.Img(src=f"data:image/png;base64,{encoded}", style={'height':'360px','margin':'6px'}), str(fpath)

# ======================================================================
# 6) Callbacks
# ======================================================================

# Tap on Cytoscape zone nodes
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
    if triggered_id == 'reset-button':
        return [], 'Selected Zone: ', elements_init
    if triggered_id == 'dash-cytoscape' and tapped_node:
        node_id = tapped_node['id']
        new_selected = selected_nodes + [node_id]
        current_nodes = [
            {"data": {"id": k, "label": k.upper()},
             "position": {"x": (v[0]+3)*100, "y": (-v[1]+3)*100},
             "locked": True, "classes": ''} for k, v in node_position.items()
        ]
        current_edges = []
        if len(new_selected) > 1:
            for i in range(len(new_selected)-1):
                source, target = new_selected[i], new_selected[i+1]
                current_edges.append({"data": {"source": source, "target": target}, "classes": "new-edge"})
        text_output = 'Selected Zone: ' + ' -> '.join(new_selected)
        return new_selected, text_output, current_nodes + current_edges
    return no_update, no_update, no_update

# Start Analysis → clustering and images
@app.callback(
    Output('image-display', 'children'),
    Output('pitch-type-coords-store', 'data'),
    Output('cluster-store', 'data'),    # trajectory / PitchSequence / SpeedSequence を格納
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

    full_df, new_pitch_coords = select_and_preprocess_data(p_name, seasons, s_type)
    if full_df.empty:
        return html.Div(f"No data found for {p_name}."), no_update, {}

    prefix_str = "".join(selected_nodes or [])
    if prefix_str:
        filtered = full_df[full_df['trajectory'].str.startswith(prefix_str)].copy()
        if filtered.empty:
            return html.Div(f"No data matches the selected prefix '{prefix_str}'."), no_update, {}
    else:
        filtered = full_df.copy()

    categories_to_process = ['All'] if ('All' in categories) else categories
    year_str = "_".join(map(str, sorted(seasons))) if isinstance(seasons, list) else str(seasons)
    seq_str = prefix_str if prefix_str else "any"

    cluster_payload = {}

    for cat in categories_to_process:
        cache_params = f"{p_name}_{year_str}_{s_type}_{cat}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
        cache_filename = f"clustered_{cache_params}.pkl"
        cache_filepath = cache_directory / cache_filename

        if cache_filepath.exists():
            try:
                with cache_filepath.open('rb') as f:
                    cat_df_with_labels = pickle.load(f)
            except Exception:
                logger.exception('Failed loading cache %s', cache_filepath)
                cat_df_with_labels = None
        else:
            cat_df_with_labels = None

        if cat_df_with_labels is None:
            if cat == 'All':
                cat_df = filtered.reset_index(drop=True)
            else:
                cat_df = filtered[filtered['Category'] == cat].reset_index(drop=True)
            if cat_df.empty: 
                continue

            labels = perform_clustering(cat_df, num_c, w_t, w_s, w_p, new_pitch_coords)
            if labels.size == 0: 
                continue

            cat_df = cat_df.copy()
            cat_df['class'] = labels
            cat_df_with_labels = cat_df

            try:
                with cache_filepath.open('wb') as f:
                    pickle.dump(cat_df_with_labels, f)
            except Exception:
                logger.exception('Failed writing cache %s', cache_filepath)

        if cat_df_with_labels.empty:
            continue

        # cluster-store へ保存（trajectory, PitchSequence, SpeedSequence）
        for i in range(int(num_c)):
            key = f"{cat}_{i}"
            subset = cat_df_with_labels[cat_df_with_labels['class'] == i]
            if subset.empty:
                continue
            records = subset[['trajectory', 'PitchSequence', 'SpeedSequence']].to_dict('records')
            cluster_payload[key] = records

        # クラスタの概観ネットワーク画像生成
        try:
            create_NX(cat_df_with_labels, p_name, year_str, s_type, seq_str, num_c, cat, node_size, min_edge)
        except Exception:
            logger.exception('create_NX failed for category %s', cat)

    image_elements = update_image_display(p_name, seasons, s_type, selected_nodes, num_c, categories, min_edge)
    return image_elements, new_pitch_coords, cluster_payload

# Cluster button → show frequency table (trajectory, count)
@app.callback(
    Output('cluster-table', 'children'),
    Input({'type': 'cluster-btn', 'category': ALL, 'cluster': ALL}, 'n_clicks'),
    State('cluster-store', 'data')
)
def show_cluster_table(n_clicks_list, cluster_store):
    if not n_clicks_list or sum([n or 0 for n in n_clicks_list]) == 0:
        return no_update
    ctx = dash.callback_context
    if not ctx.triggered: return no_update

    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    meta = json.loads(trig)
    cat = meta['category']
    idx = meta['cluster']
    key = f"{cat}_{idx}"

    records = cluster_store.get(key, [])
    if not records:
        return html.Div(f"No sequences for {cat} Cluster {idx+1}.")

    df = pd.DataFrame(records)
    freq = df.groupby('trajectory').size().reset_index(name='count').sort_values('count', ascending=False)

    content = [
        html.H4(f"{cat} - Cluster {idx+1}"),
        html.P(f"Sequences: {len(df)}  Unique trajectories: {len(freq)}"),
        dash_table.DataTable(
            id={'type': 'seqfreq-table', 'category': cat, 'cluster': idx},
            columns=[{'name': 'trajectory', 'id': 'trajectory'},
                     {'name': 'count', 'id': 'count'}],
            data=freq.to_dict('records'),
            page_size=20,
            sort_action='native',
            filter_action='native',
            cell_selectable=True,
            style_table={'overflowX': 'auto'},
            style_cell={'fontFamily':'Helvetica','fontSize':16}
        )
    ]
    return html.Div(content)

# 行クリック → PNG生成 → URLをStoreにセット（自動で別タブを開く）
@app.callback(
    Output('seq-preview', 'children'),
    Output('network-url', 'data'),
    Input({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'active_cell'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'data'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'id'),
    State('cluster-store', 'data')
)
def preview_single_sequence_from_freq(active_cells, all_tables_data, all_tables_ids, cluster_store):
    if not active_cells or sum(1 for ac in active_cells if ac) == 0:
        return no_update, no_update
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    meta = json.loads(trig)  # {'type':'seqfreq-table','category':..., 'cluster':...}
    cat = meta['category']; idx = meta['cluster']

    table_idx = None
    for i, tid in enumerate(all_tables_ids):
        if tid == meta:
            table_idx = i
            break
    if table_idx is None:
        return no_update, no_update

    ac = active_cells[table_idx]
    data = all_tables_data[table_idx]
    if not ac or 'row' not in ac or ac['row'] is None:
        return no_update, no_update

    row = data[ac['row']]
    traj = row.get('trajectory', '')
    if not traj:
        return html.Div("Invalid trajectory."), no_update

    key = f"{cat}_{idx}"
    records = cluster_store.get(key, [])
    if not records:
        return html.Div(f"No records for {cat} Cluster {idx+1}."), no_update

    cand = [r for r in records if r.get('trajectory') == traj]
    if not cand:
        return html.Div("No pitch/speed sequences found for the selected trajectory."), no_update

    rep = cand[0]
    pitch_seq = rep.get('PitchSequence', '')
    speed_seq = rep.get('SpeedSequence', '')

    buf = generate_pitchspeed_network_png(traj, pitch_seq, speed_seq)
    if not buf:
        return html.Div("Failed to generate PNG."), no_update

    cache_key_src = f"{traj}|{pitch_seq}|{speed_seq}"
    cache_key = md5(cache_key_src.encode('utf-8')).hexdigest()[:16]
    NETWORK_IMAGE_CACHE[cache_key] = buf

    url = f"/dynamic_network/{cache_key}"
    return html.Div([
        html.H4(f"Network for {traj}"),
        html.A("Open network PNG in new tab", href=url, target="_blank",
               style={'fontWeight':'bold','textDecoration':'underline'})
    ]), url

# ======================================================================
# 7) クライアントサイド：Store更新で自動で別タブを開く
# ======================================================================
app.clientside_callback(
    """
    function(url){
      if(url){ window.open(url, "_blank"); }
      return null;
    }
    """,
    Output('network-launch', 'data'),
    Input('network-url', 'data')
)

# ======================================================================
# 8) 画像配信（メモリからPNG）
# ======================================================================
@server.route("/dynamic_network/<string:key>")
def serve_dynamic_network(key):
    buf = NETWORK_IMAGE_CACHE.get(key)
    if not buf:
        return Response("Not found", status=404)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

# ======================================================================
# 9) 単一trajectory用ネットワークPNG（保存なし）
# ======================================================================
def generate_pitchspeed_network_png(traj_str, pitch_seq, speed_seq):
    """
    単一 trajectory のネットワーク図（保存無し）
    - レイアウト: node_position 固定
    - ノードサイズ: 球速 S/M/H 倍率を合算
    - ノード色: 球種カテゴリ（F=赤, S=青, C=紫, D=緑, O=灰, U=黒）
    - エッジ: 黒・不透明、太さ=頻度比例
    """
    if not traj_str or len(traj_str) < 1:
        return None

    pitch_color_map = {'F':'red','S':'blue','C':'purple','D':'green','O':'gray','U':'black'}
    speed_size_map = {'S': 0.6, 'M': 1.0, 'H': 1.4}

    zone_counts = Counter()
    zone_size_acc = Counter()
    zone_pitch_votes = {z: Counter() for z in 'abcdefghijklm'}

    steps = min(len(traj_str), len(pitch_seq), len(speed_seq))
    for i in range(steps):
        z = traj_str[i]
        p = pitch_seq[i] if i < len(pitch_seq) else 'U'
        s = speed_seq[i] if i < len(speed_seq) else 'M'
        if z not in node_position: 
            continue
        zone_counts[z] += 1
        zone_size_acc[z] += speed_size_map.get(s, 1.0)
        zone_pitch_votes[z][p] += 1

    G = nx.DiGraph()
    base_nodesize = 100

    for z, cnt in zone_counts.items():
        if zone_pitch_votes[z]:
            pitch_cat, _ = zone_pitch_votes[z].most_common(1)[0]
        else:
            pitch_cat = 'U'
        color = pitch_color_map.get(pitch_cat, 'gray')
        size = max(12, base_nodesize * zone_size_acc[z])
        G.add_node(z, color=color, size=size)

    edge_counts = Counter()
    for i in range(steps - 1):
        a, b = traj_str[i], traj_str[i+1]
        if a in zone_counts and b in zone_counts:
            edge_counts[(a, b)] += 1
    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)

    pos = {nid: ((xy[0] + 3) * 1.0, (xy[1] + 3) * 1.0) for nid, xy in node_position.items() if nid in G.nodes}

    edge_colors = ['black' for _ in G.edges()]
    edge_widths = [max(1.2, 0.8 * data.get('weight', 1)) for _, _, data in G.edges(data=True)]

    buf = io.BytesIO()
    plt.figure(figsize=(6, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=[attr['size'] for _, attr in G.nodes(data=True)],
        node_color=[attr['color'] for _, attr in G.nodes(data=True)],
        edge_color=edge_colors,
        width=edge_widths,
        font_size=12
    )
    plt.title(f"Trajectory Network (pitch/speed): {traj_str}")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close('all')
    buf.seek(0)
    return buf

# ======================================================================
# 10) Run
# ======================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
