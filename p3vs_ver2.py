import dash
import base64
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Levenshtein
import networkx as nx
from sklearn_extra.cluster import KMedoids
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

# Initialization
app = dash.Dash(__name__)
server = app.server

image_directory = '/Users/tsujistencia/Desktop/Cash' # Make sure this path is correct for your system
image_extension = '.png'

# Node and Edge initial data
nodes_init_data = [
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

edges_init_data = []
elements_init = nodes_init_data + edges_init_data

# Cytoscape component
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
        {   "selector": 'node[selected]',
            "style": {
                "border-width": 3,
                "border-color": "black"
            }
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
            "selector": ".new-edge", # Class for newly added edges including self-loops
            "style": {
                "line-color": "red",
                "target-arrow-color": "red",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                # Properties for self-loops
                "loop-direction": "0deg",  # Adjust as needed for appearance
                "loop-sweep": "90deg", # Negative sweep for a different loop direction
                "control-point-step-size": 100, # control point for bezier edges and loops
            }
        }
    ]
)

# Layout
app.layout = html.Div([
    html.Div([
        html.H1('MLB Pitcher Pitching Pattern Visualization System', style={'textAlign': 'center'})
    ], style={'width': '100%', 'margin-bottom': '20px'}),

    html.Div([
        # Left Column
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
                options=[{'value': year, 'label': str(year)} for year in range(2015, 2025)],
                multi=True,
                value=[2021, 2022, 2023],
            ),
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
                options=[{'label': str(i), 'value': str(i)} for i in range(2, 7)],
                value='3',
            )
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Middle Column
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
                value=['All']
            ),
            html.H3("Node Size Scale"),
            dcc.Slider(
                id='nodesize_slider',
                min=10, max=100, step=10, value=50,
                marks={i: str(i) for i in range(10, 100, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False
            ),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right Column
        html.Div([
            html.H3("Zone Selection", style={"textAlign": "left"}),
            html.Button('Start', id='Start-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Reset', id='reset-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Elbow', id='Elbow-button', n_clicks=0),
            cyto_compo1,
            html.Div(id='selected-zones', style={'marginTop': '20px', 'fontSize':'12px', 'minHeight':'30px'})
        ], style={'width': '25%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'})

    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center'}),

    html.Div(id='image-display', style={'width': '100%', 'textAlign': 'center', 'marginTop': '20px'}),
    dcc.Store(id='selected-nodes', data=[])
])

# Helper functions

def update_elements(tapped_node, elements, current_store_selected_nodes):
    if tapped_node is None:
        return elements, 'Selected Zone : ' + ', '.join(current_store_selected_nodes), current_store_selected_nodes

    node_id = tapped_node['id']

    if not hasattr(update_elements, 'tap_sequence_selected_nodes'):
        update_elements.tap_sequence_selected_nodes = []

    update_elements.tap_sequence_selected_nodes.append(node_id)
    
    # Remove only dynamically added edges before recalculating them
    new_elements = [el for el in elements if not (el.get('group') == 'edges' and el.get('classes') == 'new-edge')]
    # Add back nodes (or ensure elements contains all nodes if they were accidentally filtered)
    # A safer way to rebuild elements: start with nodes, then add new dynamic edges.
    # For simplicity, assuming 'elements' passed in still has nodes, and we are just replacing '.new-edge' edges.
    # A more robust way for new_elements:
    current_nodes = [el for el in elements if el.get('group') == 'nodes' or ('data' in el and 'source' not in el['data'])] # Get all node elements
    new_elements = current_nodes[:] # Start with a copy of node elements

    for i in range(len(update_elements.tap_sequence_selected_nodes) -1):
        source = update_elements.tap_sequence_selected_nodes[i]
        target = update_elements.tap_sequence_selected_nodes[i+1]
        # Create a unique ID for the edge, e.g. based on sequence index
        edge_id = f"dynamic-edge-{source}-{target}-{i}" 
        
        # *** MODIFICATION: Allow self-loops by removing the 'if source == target: continue' condition ***
        # if source == target: 
        #     continue # This line was removed to allow self-loops

        new_edge = {
            "data": {"source": source, "target": target, "id": edge_id},
            "classes": "new-edge" 
        }
        new_elements.append(new_edge)

    # Update visual selection state of nodes
    for el_idx, el_data in enumerate(new_elements):
        if 'data' in el_data and 'id' in el_data['data'] and 'source' not in el_data['data']: # It's a node
            is_selected = el_data['data']['id'] in update_elements.tap_sequence_selected_nodes
            # Ensure 'selected' key exists before trying to modify it or create it
            new_elements[el_idx]['data']['selected'] = is_selected


    selected_zones_text = 'Selected Zone : ' + ', '.join(update_elements.tap_sequence_selected_nodes)
    
    return new_elements, selected_zones_text, update_elements.tap_sequence_selected_nodes


def select_extraction_data(name, season_list, Stype, selected_nodes_list, category_item):
    path = f'/Users/tsujistencia/Desktop/Player_Data/{name}.csv'
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        return pd.DataFrame()

    try:
        data = pd.read_csv(path, usecols=['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
                                          'release_pos_y', 'release_pos_z', 'player_name', 'batter',
                                          'pitcher', 'zone', 'stand', 'p_throws', 'type', 'bb_type',
                                          'inning', 'inning_topbot', 'effective_speed', 'release_spin_rate',
                                          'game_pk', 'fielder_2', 'at_bat_number', 'pitch_number', 'pitch_name','events'])
    except Exception as e:
        print(f"Error reading CSV {path}: {e}")
        return pd.DataFrame()

    data['game_date'] = pd.to_datetime(data['game_date'])
    data['year'] = data['game_date'].dt.year
    zone_mapping = {
        1: 'c', 2: 'b', 3: 'a', 4: 'f', 5: 'e', 6: 'd',
        7: 'i', 8: 'h', 9: 'g', 11: 'k', 12: 'j', 13: 'm', 14: 'l'
    }
    data['zone'] = data['zone'].map(zone_mapping).fillna(data['zone'].astype(str))
    data.sort_values(by=['game_date', 'game_pk', 'inning', 'inning_topbot', 'at_bat_number', 'pitch_number'],
                     ascending=[True, True, True, False, True, True], inplace=True)
    
    if not isinstance(season_list, list): season_list = [season_list]
    data = data[data['year'].isin(season_list)]

    if Stype != 'All':
        data = data[data['stand'] == Stype]

    gb = data.groupby(['game_date', 'inning', 'at_bat_number'])
    trajects = []
    results = []
    for _, g in gb:
        if len(g) < 1: 
            continue
        else:
            s = ''.join(map(str, g['zone']))
            trajects.append(s)
            result = g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None
            results.append(result)

    if not trajects:
        print("No trajectories formed after grouping.")
        return pd.DataFrame()

    return_df = pd.DataFrame({'trajectory': trajects, 'Result': results})

    event_map = {
        'stolen_base_2b': 'Out', 'catcher_interf': 'Walk',
        'double_play': 'Out', 'field_out': 'Out', 'fielders_choice_out': 'Out', 'force_out': 'Out',
        'grounded_into_double_play': 'Out', 'strikeout': 'StrikeOut',
        'strikeout_double_play': 'StrikeOut',
        'double': 'BaseHit', 'hit_by_pitch': 'Walk', 'single': 'BaseHit', 'triple': 'BaseHit', 'walk': 'Walk',
        'home_run': 'HomeRun',
    }
    return_df['Category'] = return_df['Result'].map(event_map).fillna('Other')
    
    if category_item != 'All':
        return_df = return_df[return_df['Category'] == category_item]
    
    if not selected_nodes_list:
        return return_df.reset_index(drop=True)

    str_selected_nodes = [str(node) for node in selected_nodes_list]
    
    def check_trajectory(item_trajectory):
        if not isinstance(item_trajectory, str): return False
        if len(item_trajectory) < len(str_selected_nodes): return False
        return all(item_trajectory[i] == str_selected_nodes[i] for i in range(len(str_selected_nodes)))

    filtered_data = return_df[return_df['trajectory'].apply(check_trajectory)]
    return filtered_data.reset_index(drop=True)


def clustering_data(filtered_data, num_classes_str):
  if filtered_data.empty:
      return np.array([])
  num_classes = int(num_classes_str)
  trajectories = filtered_data['trajectory'].tolist()
  if len(trajectories) == 0:
      return np.array([])
  if len(trajectories) < num_classes:
      num_classes = len(trajectories)
      if num_classes == 0: return np.array([])

  dist = np.zeros((len(trajectories), len(trajectories)))
  for i in range(len(trajectories)):
    for j in range(i, len(trajectories)):
      str1 = trajectories[i]
      str2 = trajectories[j]
      d = Levenshtein.distance(str1, str2)
      dist[i][j] = d
      dist[j][i] = d
  
  if num_classes == 0 :
        return np.array([])

  kmedoids = KMedoids(n_clusters=num_classes, random_state=0, init='k-medoids++', metric='precomputed')
  try:
      kmedoids.fit(dist)
      clustered_data = kmedoids.labels_
  except Exception as e:
      print(f"Error during KMedoids fitting: {e}")
      clustered_data = np.zeros(len(trajectories), dtype=int) 
  return clustered_data


def kmedoids_fit(data, label):
  if data.empty or len(label)==0: #Added check for empty label
      return pd.DataFrame(columns=['trajects', 'Category', 'class'])
  dic = {
      'trajects': data['trajectory'],
      'Category': data['Category'],
      'class': label,
  }
  kmedoids_results = pd.DataFrame(dic)
  return kmedoids_results


def class_choice(data, num_str):
  class_data = data[data['class'] == int(num_str)]
  class_data = class_data['trajects'].values.tolist()
  return class_data


def nodeCount0(data_list_of_strings):
    nodeDic = {chr(ord('a') + i): 0 for i in range(13)} 
    valid_nodes = set(nodeDic.keys())
    for trajectory_str in data_list_of_strings:
        for node_char in trajectory_str:
            if node_char in valid_nodes:
                nodeDic[node_char] += 1
    return nodeDic


def edgeCount(data_list_of_strings):
  edgeDic = {}
  for trajectory_str in data_list_of_strings:
    for j in range(1, len(trajectory_str)):
        edge = (trajectory_str[j-1], trajectory_str[j])
        edgeDic[edge] = edgeDic.get(edge, 0) + 1
  return edgeDic


def edge_merge(edge_data_dict):
  result = {}
  for key_tuple, value_count in edge_data_dict.items():
      if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
          normalized_key = tuple(sorted(key_tuple))
          result[normalized_key] = result.get(normalized_key, 0) + value_count
  return result


def create_NX(clustered_data_df, pitcher_name, year_list, stance_type, selected_nodes_list, num_classes_str, category_item, nodesize_unit):
    year_str = "_".join(map(str, year_list)) if isinstance(year_list, list) else str(year_list)
    selected_nodes_str = "_".join(selected_nodes_list) if selected_nodes_list else "None"

    allowed_nodes = ['a','b','c','d','e','f','g','h','i','j','k','l','m']
    s_zone = set(['a','b','c','d','e','f','g','h','i'])
    b_zone = set(['j','k','l','m'])

    for i in range(int(num_classes_str)):
        G = nx.Graph()
        G.add_nodes_from([
            (n, {"color": "darkorange" if n in s_zone else "skyblue"}) for n in allowed_nodes
        ])

        class_specific_trajectories = class_choice(clustered_data_df, str(i))
        if not class_specific_trajectories:
            continue

        node_counts = nodeCount0(class_specific_trajectories)
        
        nodesize = []
        for node_id in G.nodes():
            if node_id in allowed_nodes:
                cnt = node_counts.get(node_id, 0)
                nodesize.append(nodesize_unit * cnt + nodesize_unit/5) 
            else:
                nodesize.append(nodesize_unit/5)
       
        edge_counts_raw = edgeCount(class_specific_trajectories)
        merged_edge_counts = edge_merge(edge_counts_raw)

        for k_tuple, v_weight in merged_edge_counts.items():
            # If you want self-loops in NetworkX graphs as well, remove "k_tuple[0] == k_tuple[1]" from condition
            if v_weight <= 1 or k_tuple[0] == k_tuple[1]: # Currently skips self-loops in static images
                continue
            if not (k_tuple[0] in allowed_nodes and k_tuple[1] in allowed_nodes):
                continue

            color = 'b' if k_tuple[0] in b_zone and k_tuple[1] in b_zone else \
                    'r' if k_tuple[0] in s_zone and k_tuple[1] in s_zone else 'g'
            G.add_edge(k_tuple[0], k_tuple[1], weight=v_weight, color=color)

        node_color_list = [data['color'] for node, data in G.nodes(data=True)]
        
        # Ensure G has edges before trying to access G[u][v]
        edge_weights_list = [G[u][v]['weight']/2 for u,v in G.edges()] if G.edges else []
        edge_colors_list  = [G[u][v]['color']  for u,v in G.edges()] if G.edges else []


        pos = {
            'a':[-1.2,1.5], 'b':[0, 1.9],  'c':[1.2, 1.5],
            'd':[-1.7,0],   'e':[0.1,0.3], 'f':[1.7,0],
            'g':[-1.2,-1.5],'h':[0,-1.9],  'i':[1.2,-1.5],
            'j':[-3,2],     'k':[3,2],     'l':[-3,-2], 'm':[3,-2]
        }
        
        safe_pitcher_name = "".join(c if c.isalnum() else "_" for c in pitcher_name)
        safe_category_item = "".join(c if c.isalnum() else "_" for c in category_item)
        safe_selected_nodes_str = "".join(c if c.isalnum() else "_" for c in selected_nodes_str)

        filename = f'{safe_pitcher_name}_{year_str}_{stance_type}_{safe_category_item}_{safe_selected_nodes_str}_{num_classes_str}_{i}{image_extension}'
        output_path = os.path.join(image_directory, filename)
        
        if not G.nodes:
            continue
        
        nx.draw(
            G, pos, with_labels=True,
            node_size=nodesize,
            node_color=node_color_list,
            edge_color=edge_colors_list,
            width=edge_weights_list,
            font_size=25
        )
        plt.title(f"{pitcher_name} - {category_item} - Class {i+1}/{num_classes_str}\nYear: {year_str}, Stance: {stance_type}, Seq: {selected_nodes_str}", fontsize=10)
        
        try:
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            plt.savefig(output_path)
        except Exception as e:
            print(f"Error saving image {output_path}: {e}")
        plt.close('all')


def update_NX(pitcher_name, year_list, stance_type, selected_nodes_list, num_classes_str, categories_list):
    images_html_output = []
    
    year_str = "_".join(map(str, year_list)) if isinstance(year_list, list) else str(year_list)
    selected_nodes_str = "_".join(selected_nodes_list) if selected_nodes_list else "None"

    if not categories_list:
        categories_list = []

    for category_item in categories_list:
        category_images_div_content = []
        
        for i in range(int(num_classes_str)):
            safe_pitcher_name = "".join(c if c.isalnum() else "_" for c in pitcher_name)
            safe_category_item = "".join(c if c.isalnum() else "_" for c in category_item)
            safe_selected_nodes_str = "".join(c if c.isalnum() else "_" for c in selected_nodes_str)

            filename = f'{safe_pitcher_name}_{year_str}_{stance_type}_{safe_category_item}_{safe_selected_nodes_str}_{num_classes_str}_{i}{image_extension}'
            image_path = os.path.join(image_directory, filename)
            
            if os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                    category_images_div_content.append(html.Img(src=f'data:image/png;base64,{encoded_img}',
                                                        style={'height': '200px', 'width': '200px', 'objectFit': 'contain', 'margin':'5px'}))
                except Exception as e:
                    category_images_div_content.append(html.Div(f"Error loading image: {filename}", style={'margin':'5px'}))
            else:
                category_images_div_content.append(html.Div(f"Image not found: {filename}", style={'margin':'5px'}))

        if category_images_div_content:
            category_html_block = html.Div([
                html.H4(f"Results for Category: {category_item}"),
                html.Div(category_images_div_content, style={'display': 'flex', 'flexDirection': 'row', 'flexWrap':'wrap', 'gap': '10px','justifyContent':'center'})
            ])
            images_html_output.append(category_html_block)

    return images_html_output if images_html_output else html.P("No images generated or found based on current selections.")


# Callback
@app.callback(
    [Output('selected-zones', 'children'),
     Output('image-display', 'children'),
     Output('selected-nodes', 'data'), 
     Output('dash_cyto_layout', 'elements')],
    [Input('Start-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('dash_cyto_layout', 'tapNodeData'),
     Input('nodesize_slider', 'value')],
    [State('P-name', 'value'),
     State('Season', 'value'),
     State('Stype', 'value'),
     State('numc', 'value'),
     State('Category', 'value'), 
     State('selected-nodes', 'data'), 
     State('dash_cyto_layout', 'elements')]
)
def handle_interactions(start_clicks, reset_clicks, tapped_node_data, nodesize_slider_value,
                        pitcher_name_state, season_state, stance_type_state, num_classes_state, category_state_list,
                        current_selected_nodes_from_store, current_cytoscape_elements):
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'Selected Zone : ', [], [], elements_init

    triggered_prop_id = ctx.triggered[0]['prop_id']
    triggered_element_id = triggered_prop_id.split('.')[0]

    output_selected_zones_text = dash.no_update
    output_image_display = dash.no_update
    output_selected_nodes_store = dash.no_update
    output_cytoscape_elements = dash.no_update

    if triggered_element_id == 'reset-button':
        if hasattr(update_elements, 'tap_sequence_selected_nodes'):
            delattr(update_elements, 'tap_sequence_selected_nodes') 
        return 'Selected Zone : ', [], [], elements_init

    elif triggered_element_id == 'dash_cyto_layout' and tapped_node_data is not None:
        new_cyto_elements, new_zones_text, updated_tap_sequence = update_elements(
            tapped_node_data, current_cytoscape_elements, current_selected_nodes_from_store
        )
        output_selected_zones_text = new_zones_text
        output_selected_nodes_store = updated_tap_sequence 
        output_cytoscape_elements = new_cyto_elements
        return output_selected_zones_text, output_image_display, output_selected_nodes_store, output_cytoscape_elements

    elif triggered_element_id == 'Start-button' or triggered_element_id == 'nodesize_slider':
        if not all([pitcher_name_state, season_state, stance_type_state, num_classes_state, category_state_list]):
            current_text = 'Selected Zone : ' + ', '.join(current_selected_nodes_from_store) if current_selected_nodes_from_store else 'Selected Zone : '
            return "Please ensure Pitcher, Season, Stance, Num Classes, and Category are selected.", [], current_selected_nodes_from_store, current_cytoscape_elements
        
        if not category_state_list :
             return 'No Category Selected. Please select at least one category.', [], current_selected_nodes_from_store, current_cytoscape_elements

        for cat_item in category_state_list: 
            filtered_data = select_extraction_data(pitcher_name_state, season_state, stance_type_state, current_selected_nodes_from_store, cat_item)
            
            if filtered_data.empty:
                continue

            clustered_labels = clustering_data(filtered_data, num_classes_state)
            if len(clustered_labels) == 0 and not filtered_data.empty :
                 continue

            kmedoids_results_df = kmedoids_fit(filtered_data, clustered_labels)
            if kmedoids_results_df.empty:
                continue
            
            create_NX(kmedoids_results_df, pitcher_name_state, season_state, stance_type_state, 
                      current_selected_nodes_from_store, num_classes_state, cat_item, nodesize_slider_value)

        output_image_display = update_NX(pitcher_name_state, season_state, stance_type_state, 
                                         current_selected_nodes_from_store, num_classes_state, category_state_list)
        
        output_selected_zones_text = 'Selected Zone : ' + ', '.join(current_selected_nodes_from_store if current_selected_nodes_from_store else [])
        output_selected_nodes_store = current_selected_nodes_from_store
        output_cytoscape_elements = current_cytoscape_elements

        return output_selected_zones_text, output_image_display, output_selected_nodes_store, output_cytoscape_elements

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)