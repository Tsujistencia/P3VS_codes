# ======================================================================
# 1) Imports and Configuration
# ======================================================================
import os
import io
import json
import pickle
import base64
import logging
import datetime as dt
from pathlib import Path
from hashlib import md5
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI不要なバックエンドを使用
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import networkx as nx
import Levenshtein
from sklearn_extra.cluster import KMedoids

import dash
from dash import Dash, html, dcc, no_update, dash_table, callback_context
from dash.dependencies import Input, Output, State, ALL
import dash_cytoscape as cyto
from flask import Response

# ======================================================================
# 2) Logger Configuration
# ======================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('p3vs.log', encoding='utf-8')
    ]
)

# ======================================================================
# 3) Application Constants
# ======================================================================
# Image settings
IMAGE_EXTENSION = '.png'
IMAGE_DPI = 100
IMAGE_FIGSIZE = (4, 4)
THUMBNAIL_HEIGHT = '140px'

# Clustering parameters
EDIT_DISTANCE_TRAJECTORY_COST = 2.83
EDIT_DISTANCE_SPEED_COST = 1.0
EDIT_DISTANCE_PITCH_COST = 0.5

# Data settings
DEFAULT_SEASON = 2023
STANCE_TYPES = ['All', 'L', 'R']
MIN_CLUSTER_SIZE = 2
DEFAULT_RANDOM_STATE = 0
MAX_CLUSTER_IMAGES = 60
MAX_CLUSTERING_CACHE_SIZE = 100

# Strike zone definition
STRIKE_ZONES = frozenset(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
BALL_ZONES = frozenset(['j', 'k', 'l', 'm'])
ALLOWED_ZONES = STRIKE_ZONES | BALL_ZONES

# Graph display settings
GRAPH_COLORS = {
    'strike_zone': 'darkorange',
    'ball_zone': 'skyblue',
    'edge_ball': '#1f6fe5',
    'edge_strike': '#d32f2f',
    'edge_other': '#12a36e'
}

PITCH_COLORS = {
    'F': 'red',
    'S': 'blue',
    'C': 'purple',
    'D': 'green',
    'O': 'gray',
    'U': 'black'
}

@dataclass
class TrajectoryData:
    """トラジェクトリーデータを格納するクラス"""
    trajectory: str
    coords: List[Tuple[float, float]]
    speed_sequence: str
    pitch_sequence: str
    result: Optional[str]
    category: str

@dataclass
class ClusteringResult:
    """クラスタリング結果を格納するクラス"""
    labels: np.ndarray
    medoids: List[int]
    cluster_sizes: Dict[int, int]
    distance_matrix: np.ndarray

@dataclass
class PitchTypeCoords:
    """投球タイプごとの座標情報を格納するクラス"""
    x: float
    z: float
    count: int
    speed: float

# ======================================================================
# 4) External Module Imports (with error handling)
# ======================================================================
try:
    from config import (
        EVENT_MAP, CATEGORY_MAP, RESULT_EVENT_MAP,
        NODE_POSITION, SPEED_COORDINATES
    )
    logger.debug("Successfully imported config modules")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Config module not found: {e}. Using default values.")
    EVENT_MAP = {}
    CATEGORY_MAP = {}
    RESULT_EVENT_MAP = {}
    NODE_POSITION = {}
    SPEED_COORDINATES = {}

try:
    from distance_utils import DistanceCalculator
    logger.debug("Successfully imported DistanceCalculator")
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"distance_utils module not found: {e}")
    raise

# ======================================================================
# 5) Initialize Cache and Global Variables
# ======================================================================

NETWORK_IMAGE_CACHE = {}

# ======================================================================
# 6) Clustering Cache Management
# ======================================================================
CLUSTERING_CACHE: Dict[str, np.ndarray] = {}
CLUSTERING_CACHE_MAX_SIZE = MAX_CLUSTERING_CACHE_SIZE

def get_clustering_cache_key(
    pitcher_name: str,
    year_str: str,
    stance_type: str,
    category: str,
    seq_str: str,
    num_classes: str,
    w_traj: float,
    w_speed: float,
    w_pitch: float
) -> str:
    """クラスタリング結果のキャッシュキーを生成
    
    Args:
        pitcher_name: 投手名
        year_str: 年度文字列
        stance_type: スタンス
        category: カテゴリー
        seq_str: シーケンス文字列
        num_classes: クラスター数
        w_traj: トラジェクトリーウェイト
        w_speed: スピードウェイト
        w_pitch: ピッチウェイト
        
    Returns:
        キャッシュキー（MD5ハッシュ）
    """
    key_str = f"{pitcher_name}_{year_str}_{stance_type}_{category}_{seq_str}_{num_classes}_{w_traj}_{w_speed}_{w_pitch}"
    return md5(key_str.encode()).hexdigest()

def get_cached_clustering(cache_key: str) -> Optional[np.ndarray]:
    """メモリキャッシュからクラスタリング結果を取得
    
    Args:
        cache_key: キャッシュキー
        
    Returns:
        キャッシュされたラベル配列、またはNone
    """
    if cache_key in CLUSTERING_CACHE:
        logger.debug(f"Cache hit: {cache_key}")
        return CLUSTERING_CACHE[cache_key]
    logger.debug(f"Cache miss: {cache_key}")
    return None

def cache_clustering_result(cache_key: str, labels: np.ndarray) -> None:
    """クラスタリング結果をメモリキャッシュに保存
    
    Args:
        cache_key: キャッシュキー
        labels: ラベル配列
    """
    if len(CLUSTERING_CACHE) >= CLUSTERING_CACHE_MAX_SIZE:
        # LRU削除：最初のキーを削除
        oldest_key = next(iter(CLUSTERING_CACHE))
        del CLUSTERING_CACHE[oldest_key]
        logger.debug(f"Cache evicted: {oldest_key}")
    
    CLUSTERING_CACHE[cache_key] = labels
    logger.debug(f"Cached result: {cache_key}")

# ======================================================================
# Matplotlib Utility Functions
# ======================================================================

def save_matplotlib_figure(filepath: Path, bbox_inches: str = 'tight', 
                          dpi: int = 150, pad_inches: float = 0.2) -> bool:
    """Matplotlib図を安全に保存する共通ユーティリティ関数

    Args:
        filepath: 保存先パス
        bbox_inches: bounding box設定
        dpi: 解像度
        pad_inches: パディング

    Returns:
        保存成功時True、失敗時False
    """
    try:
        plt.savefig(filepath, bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)
        logger.debug('Successfully saved matplotlib figure: %s', filepath)
        return True
    except Exception as exc:
        logger.exception('Failed to save matplotlib figure %s: %s', filepath, exc)
        return False
    finally:
        plt.close('all')

def create_matplotlib_figure_safely(figsize: Tuple[float, float] = (8, 8)) -> plt.Figure:
    """Matplotlib図を安全に作成する関数

    Args:
        figsize: 図のサイズ

    Returns:
        作成された図オブジェクト
    """
    try:
        return plt.figure(figsize=figsize)
    except Exception as exc:
        logger.exception('Failed to create matplotlib figure: %s', exc)
        return plt.figure()  # デフォルトサイズで再試行

# ======================================================================
# 1) Initialize Application and Directories
# ======================================================================

# アプリケーション設定
app = dash.Dash(__name__)
server = app.server

# パスとディレクトリ設定
HOME = Path.home()
DATA_DIRECTORY_BASE = HOME / 'Desktop/Player_Data'
IMAGE_DIRECTORY = HOME / 'Desktop/Cash'
CACHE_DIRECTORY = Path('./cache')

# キャッシュディレクトリの作成
CACHE_DIRECTORY.mkdir(exist_ok=True)

# 初期状態とレイアウト設定
INITIAL_NODES = []  # 初期選択ノード
initial_elements = []  # 初期グラフ要素
triggered_id = None  # コールバックのトリガーID
image_directory = Path(os.path.join(HOME, 'Desktop', 'Cash'))
data_directory_base = Path(os.path.join(HOME, 'Desktop', 'Player_Data'))
cache_directory = image_directory / 'Cache_CSV'

def ensure_dirs(paths: List[Path]) -> None:
    """ディレクトリが存在することを確認し、必要に応じて作成する
    
    Args:
        paths: 作成対象のPath オブジェクトのリスト
        
    Raises:
        OSError: ディレクトリ作成に失敗した場合
    """
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {p}")
        except PermissionError as e:
            logger.error(f"Permission denied for directory {p}: {e}")
            raise
        except OSError as e:
            logger.error(f"Failed to create directory {p}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating directory {p}: {e}")
            raise

ensure_dirs([image_directory, data_directory_base, cache_directory])

# ======================================================================
# 2) Initialize Distance Calculator
# ======================================================================

# ======================================================================
# 2) Helper Functions
# ======================================================================

def select_extraction_data(name: str, season: list, Stype: str, selected_nodes: list, category: str) -> pd.DataFrame:
    """投手データを抽出・フィルタリングする関数
    
    Args:
        name: 投手名
        season: シーズンのリスト
        Stype: バッターのスタンス ('All', 'L', 'R')
        selected_nodes: 選択されたゾーン
        category: 結果カテゴリー
        
    Returns:
        フィルタリング済みのDataFrame
        
    Raises:
        FileNotFoundError: CSVファイルが見つからない場合
        ValueError: データ処理エラーの場合
    """
    path = f'/Users/tsujistencia/Desktop/Player_Data/{name}.csv'
    
    # ファイルの存在確認
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Player data file not found: {path}")
    
    try:
        data = pd.read_csv(path, usecols=['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
                                          'release_pos_y', 'release_pos_z', 'player_name', 'batter',
                                          'pitcher', 'zone', 'stand', 'p_throws', 'type', 'bb_type',
                                          'inning', 'inning_topbot', 'effective_speed', 'release_spin_rate',
                                          'game_pk', 'fielder_2', 'at_bat_number', 'pitch_number', 'pitch_name','events'])
        logger.info(f"Loaded {len(data)} records from {path}")
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {path} - {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV {path}: {e}")
        raise ValueError(f"Invalid CSV format: {e}")
    except KeyError as e:
        logger.error(f"Missing required column in CSV: {e}")
        raise ValueError(f"CSV missing required column: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading CSV {path}: {e}")
        raise

    try:
        data['game_date'] = pd.to_datetime(data['game_date'], errors='coerce')
        data['year'] = data['game_date'].dt.year
        
        zone_mapping = {
            1: 'c', 2: 'b', 3: 'a', 4: 'f', 5: 'e', 6: 'd',
            7: 'i', 8: 'h', 9: 'g', 11: 'k', 12: 'j', 13: 'm', 14: 'l'
        }
        data['zone'] = data['zone'].map(zone_mapping).fillna(data['zone'])
        data.sort_values(by=['game_date', 'game_pk', 'inning', 'inning_topbot', 'at_bat_number', 'pitch_number'],
                         ascending=[True, True, True, False, True, True], inplace=True)
        
        logger.debug(f"Data before filtering: {len(data)} records")
        
        # シーズンフィルタ
        if not isinstance(season, list) or not season:
            logger.warning("Invalid season list, using default [2023]")
            season = [2023]
        data = data[data['year'].isin(season)]
        
        # スタンスフィルタ
        if Stype != 'All':
            if Stype not in ['L', 'R']:
                logger.warning(f"Invalid stance {Stype}, using 'All'")
                Stype = 'All'
            else:
                data = data[data['stand'] == Stype]
        
        # トラジェクトリー抽出
        gb = data.groupby(['game_date', 'inning', 'at_bat_number'])
        trajects = []
        results = []
        for i, g in gb:
            if len(g) < 2:
                continue
            s = ''.join(map(str, g['zone']))
            trajects.append(s)
            result = g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None
            results.append(result)

        return_df = pd.DataFrame({
            'trajectory': trajects,
            'Result': results,
        })
        
        if return_df.empty:
            logger.warning(f"No trajectories found for {name} in seasons {season}, stance {Stype}")
            return return_df

        event_map = {
            'stolen_base_2b': 'Out', 'catcher_interf': 'Walk',
            'double_play': 'Out', 'field_out': 'Out', 'fielders_choice_out': 'Out', 'force_out': 'Out',
            'grounded_into_double_play': 'Out', 'strikeout': 'StrikeOut',
            'strikeout_double_play': 'StrikeOut',
            'double': 'BaseHit', 'hit_by_pitch': 'Walk', 'single': 'BaseHit', 'triple': 'BaseHit', 'walk': 'Walk',
            'home_run': 'HomeRun',
        }

        return_df['Category'] = return_df['Result'].map(event_map).fillna(return_df['Result'])
        logger.debug(f"Data after category mapping: {len(return_df)} records")

        if category == 'All':
            return return_df
        else:
            return_df1 = return_df[return_df['Category'].isin([category])]
            logger.debug(f"Data after category filter: {len(return_df1)} records")

            if not selected_nodes or len(selected_nodes) == 0:
                return return_df1.reset_index(drop=True)

            filtered_data = return_df1[return_df1['trajectory'].apply(
                lambda item: len(item) >= len(selected_nodes) and all(item[i] == selected_nodes[i] for i in range(len(selected_nodes)))
            )]

            logger.debug(f"Data after prefix filter: {len(filtered_data)} records")
            return filtered_data.reset_index(drop=True)
            
    except ValueError as e:
        logger.error(f"Value error during data processing: {e}")
        raise
    except KeyError as e:
        logger.error(f"Key error during grouping: {e}")
        raise ValueError(f"Missing expected column during grouping: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during data processing: {e}")
        raise

def clustering_data(filtered_data: pd.DataFrame, num_classes: int) -> np.ndarray:
    """クラスタリングを実行する関数

    Args:
        filtered_data (pd.DataFrame): トラジェクトリーデータを含むDataFrame
        num_classes (int): クラスター数

    Returns:
        np.ndarray: クラスタリング結果のラベル配列。エラー時は空配列を返す
        
    Raises:
        ValueError: num_classes が無効な場合
    """
    if filtered_data.empty:
        logger.warning("Empty DataFrame provided to clustering_data")
        return np.array([])

    try:
        num_classes = int(num_classes)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid num_classes value: {num_classes} - {e}")
        raise ValueError(f"num_classes must be an integer, got {num_classes}")
    
    trajectories = filtered_data['trajectory'].tolist()
    n = len(trajectories)

    # クラスター数の調整
    if n < num_classes:
        logger.warning(f"num_classes ({num_classes}) > data size ({n}), adjusting to {n}")
        num_classes = n
    if num_classes <= 0:
        logger.error(f"Invalid num_classes after adjustment: {num_classes}")
        return np.array([])

    # 距離行列の計算
    try:
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):  # 対称行列なので上三角部分のみ計算
                try:
                    d = Levenshtein.distance(trajectories[i], trajectories[j])
                    dist[i, j] = dist[j, i] = d
                except TypeError as e:
                    logger.error(f"Error computing distance between trajectories {i} and {j}: {e}")
                    raise
        logger.debug(f"Distance matrix computed: shape={dist.shape}")
    except Exception as e:
        logger.error(f"Error during distance matrix computation: {e}")
        return np.array([])

    # KMedoidsクラスタリングの実行
    try:
        kmedoids = KMedoids(
            n_clusters=num_classes,
            random_state=DEFAULT_RANDOM_STATE,
            init='k-medoids++',
            metric='precomputed'
        )
        kmedoids.fit(dist)
        logger.info(f"KMedoids clustering completed: {num_classes} clusters, {n} samples")
        return kmedoids.labels_
    except ValueError as e:
        logger.error(f"ValueError during KMedoids clustering: {e}")
        return np.array([])
    except RuntimeError as e:
        logger.error(f"RuntimeError during KMedoids clustering: {e}")
        return np.array([])
    except Exception as e:
        logger.error(f"Unexpected error during KMedoids clustering: {e}")
        return np.array([])

def kmedoids_fit(data: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """クラスタリング結果をDataFrameとして整形する関数

    Args:
        data (pd.DataFrame): 元のデータフレーム
        labels (np.ndarray): クラスタリングで得られたラベル配列

    Returns:
        pd.DataFrame: クラスタリング結果を含むDataFrame
    """
    if data.empty or len(labels) == 0:
        return pd.DataFrame()

    result_df = pd.DataFrame({
        'trajects': data['trajectory'],
        'Category': data['Category'],
        'class': labels
    })
    
    logger.debug(f"Created clustering results DataFrame with shape: {result_df.shape}")
    return result_df

def class_choice(data: pd.DataFrame, num: Union[int, str]) -> List[str]:
    """クラスター番号に基づいてトラジェクトリーデータを抽出する関数

    Args:
        data: クラスタリング結果を含むDataFrame
        num: クラスター番号

    Returns:
        該当クラスターのトラジェクトリー文字列のリスト
    """
    class_data = data[data['class'].isin([int(num)])]
    class_data = class_data['trajects']
    class_data = class_data.values.tolist()
    return class_data

# ゾーンの定義

# ゾーンの定義（グローバル定数）
STRIKE_ZONES = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
BALL_ZONES = set(['j', 'k', 'l', 'm'])
ALLOWED_ZONES = STRIKE_ZONES | BALL_ZONES

# グローバル変数の定義
allowed_nodes = list('abcdefghijklm')
s_zone = list('abcdefghi')
b_zone = list('jklm')
image_extension = '.png'
Cash = os.path.join(HOME, 'Desktop', 'Cash')

def create_network_visualization(
    clustered_data: pd.DataFrame,
    pitcher_name: str,
    year: str,
    stance_type: str,
    selected_zones: str,
    num_classes: int,
    category: str,
    nodesize_unit: float,
) -> None:
    """クラスタリング結果をネットワーク図として可視化する関数

    Args:
        clustered_data: クラスタリング結果のDataFrame
        pitcher_name: 投手名
        year: 年度
        stance_type: 打者のスタンス
        selected_zones: 選択されたゾーン
        num_classes: クラスター数
        category: カテゴリー
        nodesize_unit: ノードサイズの単位

    Notes:
        - 結果は画像ファイルとして保存される
        - キャッシュにも保存される
    """

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
            logger.debug(f'File {update_Path} already exists. Skipping plot.')
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

def update_NX(pitcher_name: str, year: Union[str, int], stance_type: str, 
              selected_zones: List[str], num_classes: Union[str, int], 
              categories: List[str]) -> List[html.Div]:
    """ネットワーク画像の表示を更新する関数

    Args:
        pitcher_name: 投手名
        year: 年度
        stance_type: 打者のスタンス
        selected_zones: 選択されたゾーン
        num_classes: クラスター数
        categories: カテゴリーリスト

    Returns:
        HTML要素のリスト
    """
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

# Initialize distance calculator
distance_calculator = DistanceCalculator(NODE_POSITION, SPEED_COORDINATES)

def weighted_edit_distance(s1: str, s2: str, cost_func, ins_del_cost: float = 1.0) -> float:
    return distance_calculator.weighted_edit_distance(s1, s2, cost_func, ins_del_cost)

def get_trajectory_cost(c1: str, c2: str, node_pos=NODE_POSITION) -> float:
    return distance_calculator.get_trajectory_cost(c1, c2)

def get_speed_cost(c1: str, c2: str, speed_pos=SPEED_COORDINATES) -> float:
    return distance_calculator.get_speed_cost(c1, c2)

def get_pitch_type_cost(c1: str, c2: str, pitch_coords: Dict[str, Tuple[float, float]]) -> float:
    return distance_calculator.get_pitch_type_cost(c1, c2, pitch_coords)

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    return distance_calculator.normalize_matrix(matrix)

# ======================================================================
# 3) Layout
# ======================================================================

# Initialize cytoscape nodes
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

app.layout = html.Div([
    html.H1("MLB Pitcher's Pitching Patterns Visualization System", style={'textAlign': 'center', 'fontFamily': 'Helvetica'}),
    dcc.Store(id='selected-nodes-store', data=[]),
    dcc.Store(id='pitch-type-coords-store', data={}),
    dcc.Store(id='cluster-store', data={}),   # {'Cat_i':[{'trajectory','PitchSequence','SpeedSequence'}]}
    dcc.Store(id='cluster-bounds-store', data={}),
    dcc.Store(id='network-url'),
    dcc.Store(id='network-launch'),
    html.Div([
        html.Div([
            html.H3("Pitcher & Data Selection"),
            dcc.Dropdown(id='P-name', options=[
                {'label': 'Shohei Ohtani', 'value': 'Ohtani'},
                {'label': 'Yu Darvish', 'value': 'Darvish'},
                {'label': 'Yoshinobu Yamamoto', 'value': 'Yamamoto'},
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
            html.Label("Speed (S M H) Weight"),
            dcc.Slider(id='w-speed-slider', min=0, max=1, step=0.5, value=1, marks={0:'0',0.5:'0.5',1:'1'}),
            html.Label("Pitch Type Weight"),
            dcc.Slider(id='w-pitch-slider', min=0, max=1, step=0.5, value=1, marks={0:'0',0.5:'0.5',1:'1'}),
        ], style={'width': '20%', 'padding': '10px', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3("Zone Selection"),
            html.Div(id='selected-zones-text', style={'minHeight': '20px', 'fontWeight': 'bold'}),
            cyto.Cytoscape(
                id="dash_cyto_layout",
                style={"width": "100%", "height": "275px"},
                layout={"name": "preset"},
                elements=elements_init,
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {"content": "data(label)",
                                  "width": "75px",
                                  "height": "75px"},
                    },
                    {
                        "selector": "edge",
                        "style": {"width": 20, "content": "data(weight)"},
                    },
                    {
                        "selector": 'node[id ^= "a"], node[id ^= "b"], node[id ^= "c"], node[id ^= "d"], node[id ^= "e"], node[id ^= "f"], node[id ^= "g"], node[id ^= "h"], node[id ^= "i"]',
                        "style": {"background-color": "orange"}
                    },
                    {
                        "selector": 'node[id ^= "j"], node[id ^= "k"], node[id ^= "l"], node[id ^= "m"]',
                        "style": {"background-color": "lightblue"}
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
        ], style={'width': '20%', 'padding': '5px', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'flexDirection': 'row'}),

    html.Hr(),
    dcc.Loading(id="loading-spinner", children=[html.Div(id='image-display')], type="circle"),

    html.Div(id='cluster-table', style={'marginTop': '20px'}),
    html.Div(id='seq-preview', style={'marginTop': '16px'})
])

# ======================================================================
# 4) Data Processing and Clustering
# ======================================================================

def select_and_preprocess_data(name: str, season_list: list, Stype: str) -> tuple[pd.DataFrame, dict]:
    """指定された投手のデータを読み込み、前処理を行う関数

    Args:
        name (str): 投手名
        season_list (list): シーズンのリスト
        Stype (str): バッターのスタンス（'All', 'L', 'R'）

    Returns:
        tuple[pd.DataFrame, dict]: 
            - 処理済みのDataFrame
            - 投球タイプの座標情報を含む辞書
    """
    # データファイルの読み込み
    path = data_directory_base / f'{name}.csv'
    if not path.exists():
        logger.info('Data file not found: %s', path)
        return pd.DataFrame(), {}
    
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        logger.exception('Failed reading CSV %s: %s', path, exc)
        return pd.DataFrame(), {}

    # 日付とシーズンの処理
    data['game_date'] = pd.to_datetime(data['game_date'])
    data['year'] = data['game_date'].dt.year
    
    # シーズンリストのバリデーション
    if not isinstance(season_list, list) or not season_list:
        season_list = [2023]
        logger.warning('Invalid season_list provided, using default [2023]')
    
    # データのフィルタリング
    data = data[data['year'].isin(season_list)]
    if Stype != 'All':
        data = data[data['stand'] == Stype]
    
    if data.empty:
        logger.info(f'No data found for {name} in seasons {season_list} with stance {Stype}')
        return pd.DataFrame(), {}

    data['Pitch_Category_T'] = data['pitch_name'].map(EVENT_MAP).fillna(data['pitch_name'])
    data['PitchCategory'] = data['Pitch_Category_T'].map(CATEGORY_MAP).fillna('U')

    all_speeds = data['effective_speed'].dropna()
    low, high = np.percentile(all_speeds, [33, 66]) if len(all_speeds) >= 3 else (80, 90)
    def categorize_speed(speed):
        if pd.isna(speed):
            return 'M'
        if speed <= low:
            return 'S'
        elif speed <= high:
            return 'M'
        else:
            return 'H'
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
                # plate_x の符号を反転して X 軸を調整
                'coords': [( -float(x) if pd.notna(x) else x, float(z) if pd.notna(z) else z ) for x, z in zip(g['plate_x'], g['plate_z'])],
                'SpeedSequence': ''.join(g['CategorizedSpeed']),
                'PitchSequence': ''.join(g['PitchCategory']),
                'Result': g['events'].dropna().iloc[-1] if not g['events'].dropna().empty else None
            })
    if not rows:
        return pd.DataFrame(), {}
    processed_df = pd.DataFrame(rows)
    processed_df['Category'] = processed_df['Result'].map(RESULT_EVENT_MAP).fillna('Other')
    return processed_df, pitch_type_coords

def perform_clustering(df: pd.DataFrame, num_classes_str: Union[str, int], 
                      w_traj: float, w_speed: float, w_pitch: float, 
                      pitch_coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """クラスタリングを実行する関数

    Args:
        df: 入力データフレーム
        num_classes_str: クラスター数
        w_traj: トラジェクトリー重み
        w_speed: スピード重み
        w_pitch: 球種重み
        pitch_coords: 球種座標辞書

    Returns:
        クラスタリング結果のラベル配列
    """
    trajectories, n, num_classes = df.to_dict('records'), len(df), int(num_classes_str)
    if n < num_classes:
        num_classes = n
    if num_classes == 0:
        return np.array([])
    dist_traj, dist_speed, dist_pitch = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist_traj[i, j] = dist_traj[j, i] = weighted_edit_distance(
                trajectories[i]['trajectory'], trajectories[j]['trajectory'],
                get_trajectory_cost, ins_del_cost=2.83
            )
            dist_speed[i, j] = dist_speed[j, i] = weighted_edit_distance(
                trajectories[i]['SpeedSequence'], trajectories[j]['SpeedSequence'],
                get_speed_cost, ins_del_cost=1.0
            )
            cost_func = lambda c1, c2: get_pitch_type_cost(c1, c2, pitch_coords)
            dist_pitch[i, j] = dist_pitch[j, i] = weighted_edit_distance(
                trajectories[i]['PitchSequence'], trajectories[j]['PitchSequence'],
                cost_func, ins_del_cost=0.5
            )
    final_dist = (w_traj * normalize_matrix(dist_traj)
                  + w_speed * normalize_matrix(dist_speed)
                  + w_pitch * normalize_matrix(dist_pitch))
    kmedoids = KMedoids(n_clusters=num_classes, random_state=0, metric='precomputed', init='k-medoids++')
    try:
        kmedoids.fit(final_dist)
        return kmedoids.labels_
    except Exception as e:
        logger.exception("KMedoids fitting error: %s", e)
        return np.array([])

# --- helpers for cluster images ---
def class_choice(clustered_data: pd.DataFrame, class_label: int) -> pd.Series:
    """指定されたクラスのトラジェクトリーを抽出する関数

    Args:
        clustered_data: クラスタリング済みのデータフレーム
        class_label: クラスラベル

    Returns:
        該当クラスのトラジェクトリー系列
    """
    return clustered_data[clustered_data['class'] == class_label]['trajectory']

def nodeCount0(trajectories: pd.Series) -> Counter:
    """トラジェクトリー内の各文字の出現回数をカウントする関数

    Args:
        trajectories: トラジェクトリーの系列

    Returns:
        各文字の出現回数を格納したCounter
    """
    return Counter(char for traj in trajectories for char in traj)

def edgeCount(trajectories: pd.Series) -> Counter:
    """トラジェクトリー内のエッジ（連続する文字のペア）をカウントする関数

    Args:
        trajectories: トラジェクトリーの系列

    Returns:
        エッジの出現回数を格納したCounter
    """
    return Counter(edge for traj in trajectories if len(traj) > 1 for edge in zip(traj[:-1], traj[1:]))

def edge_merge(sorted_edges_with_counts: List[Tuple[Tuple[str, str], int]]) -> Dict[Tuple[str, str], int]:
    """エッジのカウントを統合する関数

    Args:
        sorted_edges_with_counts: エッジとそのカウントのペアのリスト

    Returns:
        統合されたエッジカウントの辞書
    """
    merged = {}
    for edge, count in sorted_edges_with_counts:
        canonical = tuple(sorted(edge))
        merged[canonical] = merged.get(canonical, 0) + count
    return merged

# ======================================================================
# 4.5) Cluster Overview Network
# ======================================================================
def create_NX(clustered_data, pitcher_name, year_str, stance_type, seq_str,
              num_classes, category, nodesize_unit, min_edge_weight):
    logger.info(f"create_NX called with min_edge_weight={min_edge_weight}")
    allowed_nodes = list('abcdefghijklm')
    s_zone = list('abcdefghi')
    b_zone = list('jklm')
    
    # クリーンアップ：この投手/年/スタンス/カテゴリの古い画像を削除
    old_image_pattern = f'{pitcher_name}_{year_str}_{stance_type}_{category}_*_*.png'
    logger.info(f"Cleaning up old images matching: {old_image_pattern}")
    try:
        from pathlib import Path
        for old_file in image_directory.glob(f'{pitcher_name}_{year_str}_{stance_type}_{category}_*_*.png'):
            try:
                old_file.unlink()
                logger.info(f"Deleted old image: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {old_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to clean up old images: {e}")

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
            if cnt < min_edge_weight or u == v:
                continue
            if (u not in allowed_nodes) or (v not in allowed_nodes):
                continue

            if (u in b_zone) and (v in b_zone):
                base_color = '#1f6fe5'
            elif (u in s_zone) and (v in s_zone):
                base_color = '#d32f2f'
            else:
                base_color = '#12a36e'
            G.add_edge(u, v, weight=cnt, base_color=base_color)

        pos = {
            'a':[-1.2,1.5], 'b':[0,1.9], 'c':[1.2,1.5],
            'd':[-1.7,0],   'e':[0.1,0.3], 'f':[1.7,0],
            'g':[-1.2,-1.5],'h':[0,-1.9],  'i':[1.2,-1.5],
            'j':[-3,2],     'k':[3,2],     'l':[-3,-2], 'm':[3,-2]
        }

        max_w = max((data.get('weight', 1) for _, _, data in G.edges(data=True)), default=1)

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
        logger.info(f"Generating image: {filename} with min_edge_weight={min_edge_weight}")
        if update_Path.exists():
            logger.info('File already exists, skipping: %s', filename)
            continue

        create_matplotlib_figure_safely((8, 8))
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
        save_matplotlib_figure(update_Path)

# ======================================================================
# 5) Helper UIs
# ======================================================================

def update_image_display(pitcher_name: str, seasons: List[int], stance_type: str, 
                        selected_zones: Optional[List[str]], num_classes: Union[str, int], 
                        categories: List[str], min_edge_weight: Union[str, int, float]) -> html.Div:
    """画像表示を更新する関数

    Args:
        pitcher_name: 投手名
        seasons: シーズンのリスト
        stance_type: 打者のスタンス
        selected_zones: 選択されたゾーン
        num_classes: クラスター数
        categories: カテゴリーリスト
        min_edge_weight: 最小エッジ重み

    Returns:
        更新された画像表示のHTML要素
    """
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

def draw_single_sequence_image(traj_str: str) -> Tuple[html.Div, Optional[str]]:
    """単一のトラジェクトリー画像を描画する関数

    Args:
        traj_str: トラジェクトリー文字列

    Returns:
        画像要素とファイルパス（失敗時はNone）のタプル
    """
    if not traj_str or len(traj_str) < 2:
        return html.Div("Select one trajectory row."), None
    key = md5(traj_str.encode('utf-8')).hexdigest()[:12]
    fname = f"seq_{key}.png"
    fpath = image_directory / fname
    if not fpath.exists():
        pos = {k: ((v[0]+3)*1.0, (v[1]+3)*1.0) for k, v in NODE_POSITION.items()}
        plt.figure(figsize=(6, 6))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        
        for nid, (x, y) in pos.items():
            color = 'orange' if nid in list('abcdefghi') else 'lightblue'
            ax.scatter(x, y, s=300, c=color, edgecolors='black')
            ax.text(x, y, nid.upper(), ha='center', va='center', fontsize=10, color='black')
            
        for i in range(len(traj_str)-1):
            s, t = traj_str[i], traj_str[i+1]
            if s not in pos or t not in pos:
                continue
            x1, y1 = pos[s]; x2, y2 = pos[t]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
            xm, ym = (x1+x2)/2, (y1+y2)/2
            ax.text(xm, ym, str(i+1), fontsize=9)
            
        ax.axis('off')
        # マージンを追加してオーバーラップを防ぐ
        plt.margins(0.1)
        try:
            plt.savefig(fpath, bbox_inches='tight', dpi=150, pad_inches=0.2)
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

@app.callback(
    [Output('selected-zones-text', 'children'),
     Output('selected-nodes-store', 'data'),
     Output('dash_cyto_layout', 'elements')],
    [
        Input('start-button', 'n_clicks'),
        Input('reset-button', 'n_clicks'),
        Input('dash_cyto_layout', 'tapNodeData'),
    ],
    [
        State('P-name', 'value'),
        State('Season', 'value'),
        State('Stype', 'value'),
        State('numc', 'value'),
        State('Category', 'value'),
        State('selected-nodes-store', 'data'),
        State('dash_cyto_layout', 'elements')
    ]
)
def handle_buttons(start_clicks: Optional[int], reset_clicks: Optional[int], 
                  tapped_node: Optional[Dict[str, Any]], pitcher_name: str, 
                  year: Union[List[int], int], stance_type: str, 
                  num_classes: Union[str, int], Category: List[str], 
                  selected_nodes: Optional[List[str]], 
                  elements: List[Dict[str, Any]]) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """ボタンクリックとノードタップを処理するコールバック関数

    Args:
        start_clicks: 開始ボタンのクリック数
        reset_clicks: リセットボタンのクリック数
        tapped_node: タップされたノードデータ
        pitcher_name: 投手名
        year: 年度
        stance_type: 打者のスタンス
        num_classes: クラスター数
        Category: カテゴリーリスト
        selected_nodes: 選択されたノード
        elements: グラフ要素

    Returns:
        選択ゾーンテキスト、選択ノードリスト、グラフ要素のタプル
    """
    # This implementation is transplanted from p3vs_ver9_new.py and adapted to use
    # the existing stores and variables in p3vs_ver9.py. It handles Start/Reset
    # button clicks and cytoscape node taps, updating elements and selected nodes.
    nodesize_unit = 50
    ctx = dash.callback_context

    selected_zones = 'Selected Zone：'  
    output_nodes = selected_nodes
    output_elements = elements

    if not ctx.triggered:
        return selected_zones, [], elements_init

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button' and reset_clicks > 0:
        # clear persisted selection on the helper function if present
        if hasattr(update_elements, 'selected_nodes'):
            delattr(update_elements, 'selected_nodes')
        return selected_zones, [], elements_init

    if button_id == 'dash_cyto_layout' and tapped_node is not None:
        # Delegate tap handling to update_elements helper and get updated elements/store
        output_elements, selected_zones, output_nodes = update_elements(
            tapped_node, elements=elements, selected_nodes=selected_nodes
        )

    return selected_zones, output_nodes, output_elements

@app.callback(
    Output('image-display', 'children'),
    Output('pitch-type-coords-store', 'data'),
    Output('cluster-store', 'data'),
    Output('cluster-bounds-store', 'data'),
    Input('start-button', 'n_clicks'),
    State('P-name', 'value'), State('Season', 'value'), State('Stype', 'value'),
    State('Category', 'value'), State('numc', 'value'),
    State('w-traj-slider', 'value'), State('w-speed-slider', 'value'), State('w-pitch-slider', 'value'),
    State('nodesize-slider', 'value'), State('min-edge-weight-slider', 'value'),
    State('selected-nodes-store', 'data'),
    State('pitch-type-coords-store', 'data'),
    prevent_initial_call=True
)
def run_analysis(n_clicks: int, p_name: str, seasons: List[int], s_type: str, 
                categories: List[str], num_c: Union[str, int],
                w_t: float, w_s: float, w_p: float, node_size: int, min_edge: int,
                selected_nodes: Optional[List[str]], 
                pitch_coords: Dict[str, Tuple[float, float]]) -> Tuple[html.Div, Dict[str, Tuple[float, float]], Dict[str, Any], Dict[str, Any]]:
    """分析を実行するコールバック関数

    Args:
        n_clicks: ボタンのクリック数
        p_name: 投手名
        seasons: シーズンのリスト
        s_type: 打者のスタンス
        categories: カテゴリーリスト
        num_c: クラスター数
        w_t: トラジェクトリー重み
        w_s: スピード重み
        w_p: 球種重み
        node_size: ノードサイズ
        min_edge: 最小エッジ重み
        selected_nodes: 選択されたノード
        pitch_coords: 球種座標辞書

    Returns:
        画像要素、球種座標、クラスターデータ、クラスター境界のタプル
    """

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
    cluster_bounds = {}

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
                # キャッシュに coords 列がない場合は、元の full_df から補完する
                if cat_df_with_labels is not None and 'coords' not in getattr(cat_df_with_labels, 'columns', []):
                    try:
                        traj_to_coords = dict(zip(full_df['trajectory'], full_df.get('coords', [])))
                        cat_df_with_labels = cat_df_with_labels.copy()
                        cat_df_with_labels['coords'] = cat_df_with_labels['trajectory'].map(traj_to_coords)
                    except Exception:
                        logger.exception('Failed to restore coords into cached cluster dataframe %s', cache_filepath)
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

        for i in range(int(num_c)):
            key = f"{cat}_{i}"
            subset = cat_df_with_labels[cat_df_with_labels['class'] == i]
            if subset.empty:
                continue
            records = subset[['trajectory', 'coords', 'PitchSequence', 'SpeedSequence']].to_dict('records')
            cluster_payload[key] = records
            # compute bounds for this cluster from coords
            try:
                all_coords = [c for coords_list in subset['coords'].dropna() for c in coords_list]
                xs = [float(c[0]) for c in all_coords if c is not None and pd.notna(c[0])]
                ys = [float(c[1]) for c in all_coords if c is not None and pd.notna(c[1])]
                if xs and ys:
                    cluster_bounds[key] = {'xmin': min(xs), 'xmax': max(xs), 'ymin': min(ys), 'ymax': max(ys)}
                else:
                    cluster_bounds[key] = None
            except Exception:
                logger.exception('Failed computing bounds for cluster %s', key)
                cluster_bounds[key] = None

        try:
            create_NX(cat_df_with_labels, p_name, year_str, s_type, seq_str, num_c, cat, node_size, min_edge)
        except Exception:
            logger.exception('create_NX failed for category %s', cat)
    # --- Debug: sample cluster_payload contents (helps verify coords present) ---
    try:
        if cluster_payload:
            sample_key = next(iter(cluster_payload))
            sample_list = cluster_payload.get(sample_key, [])
            sample_record = sample_list[0] if sample_list else None
            logger.info('cluster_payload: n_keys=%d, sample_key=%s, sample_record_keys=%s',
                        len(cluster_payload), sample_key, list(sample_record.keys()) if isinstance(sample_record, dict) else None)
        else:
            logger.info('cluster_payload is empty')
    except Exception:
        logger.exception('Failed while logging cluster_payload sample')

    image_elements = update_image_display(p_name, seasons, s_type, selected_nodes, num_c, categories, min_edge)
    return image_elements, new_pitch_coords, cluster_payload, cluster_bounds

@app.callback(
    Output('cluster-table', 'children'),
    Input({'type': 'cluster-btn', 'category': ALL, 'cluster': ALL}, 'n_clicks'),
    State('cluster-store', 'data')
)
def show_cluster_table(n_clicks_list: List[Optional[int]], 
                      cluster_store: Dict[str, Any]) -> Union[html.Div, dash._utils.AttributeDict]:
    """クラスターテーブルを表示するコールバック関数

    Args:
        n_clicks_list: クリック数のリスト
        cluster_store: クラスターストアデータ

    Returns:
        テーブル表示のHTML要素またはno_update
    """
    if not n_clicks_list or sum([n or 0 for n in n_clicks_list]) == 0:
        return no_update
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    meta = json.loads(trig)
    cat = meta['category']
    idx = meta['cluster']
    key = f"{cat}_{idx}"

    records = cluster_store.get(key, [])
    if not records:
        return html.Div(f"No sequences for {cat} Cluster {idx+1}.")

    df = pd.DataFrame(records)
    df.dropna(subset=['trajectory'], inplace=True)
    df.reset_index(drop=True, inplace=True)
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

@app.callback(
    Output('seq-preview', 'children'),
    Output('network-url', 'data'),
    Input({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'active_cell'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'data'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'id'),
    State('cluster-store', 'data'),
    State('cluster-bounds-store', 'data')
)
def preview_single_sequence_from_freq(active_cells, all_tables_data, all_tables_ids, cluster_store, cluster_bounds_store):
    if not active_cells or sum(1 for ac in active_cells if ac) == 0:
        return no_update, no_update
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    meta = json.loads(trig)
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
        return html.Div("No pitch or speed sequences found for the selected trajectory."), no_update

    rep = cand[0]
    traj = rep.get('trajectory', '')
    coords = rep.get('coords', [])  # plate_x, plate_zの座標のリスト
    pitch_seq = rep.get('PitchSequence', '')
    speed_seq = rep.get('SpeedSequence', '')

    if not coords or len(coords) < 2:
        return html.Div(f"シーケンス {traj} は描画に必要な座標データがありません。"), no_update

    # pitch_seq と speed_seq の処理
    if not pitch_seq:
        pitch_seq = 'F' * len(coords)  # デフォルトはすべてFastball
    
    if not speed_seq:
        speed_seq = [90.0 + i for i in range(len(coords))]  # ダミーの速度データ
    else:
        try:
            speed_seq = [float(s) for s in speed_seq]
        except (ValueError, TypeError):
            speed_seq = [90.0 + i for i in range(len(coords))]  # 変換に失敗した場合はダミーデータ

    # Create a temporary DataFrame for visualization
    df_vis = pd.DataFrame({
        'coords': [coords],
        'PitchCategory': [pitch_seq],
        'PitchSpeed': [speed_seq]
    })

    # Create figure and plot
    try:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        # if cluster bounds exist for this cluster, use as xlim/ylim with small padding
        bounds = cluster_bounds_store.get(key) if cluster_bounds_store else None
        if bounds:
            xlim = (bounds['xmin'] - 0.1 * abs(bounds['xmax'] - bounds['xmin']),
                    bounds['xmax'] + 0.1 * abs(bounds['xmax'] - bounds['xmin']))
            ylim = (bounds['ymin'] - 0.1 * abs(bounds['ymax'] - bounds['ymin']),
                    bounds['ymax'] + 0.1 * abs(bounds['ymax'] - bounds['ymin']))
            plot_seq_matplotlib2(df_vis, pos=0, ax=ax, xlim=xlim, ylim=ylim)
        else:
            plot_seq_matplotlib2(df_vis, pos=0, ax=ax)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, pad_inches=0.2)
        plt.close('all')
        buf.seek(0)

        cache_key_src = f"{traj}|{pitch_seq}|{speed_seq}"
        cache_key = md5(cache_key_src.encode('utf-8')).hexdigest()[:16]
        NETWORK_IMAGE_CACHE[cache_key] = buf

        url = f"/dynamic_network/{cache_key}"
        return html.Div([
            html.H4(f"投球シーケンス: {traj.upper()}"),
            html.P(f"球種: {pitch_seq}"),
            html.A("新しいタブで投球シーケンス図を開く", href=url, target="_blank",
                   style={'fontWeight':'bold','textDecoration':'underline'})
        ]), url
    except Exception as e:
        plt.close('all')
        return html.Div(f"エラーが発生しました: {str(e)}"), no_update

# ======================================================================
# 7) Client side window open
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
# 8) Serve in memory PNG
# ======================================================================
@server.route("/dynamic_network/<string:key>")
def serve_dynamic_network(key):
    buf = NETWORK_IMAGE_CACHE.get(key)
    if not buf:
        return Response("Not found", status=404)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

# NETWORK_IMAGE_CACHE のクリーンアップ（定期的に古いキャッシュを削除）
def cleanup_image_cache():
    NETWORK_IMAGE_CACHE.clear()

# ======================================================================
# 9) Single trajectory network PNG generator
# ======================================================================
def generate_pitchspeed_network_png(traj_str, pitch_seq, speed_seq):
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
        if z not in NODE_POSITION:
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

    pos = {nid: ((xy[0] + 3) * 1.0, (xy[1] + 3) * 1.0) for nid, xy in NODE_POSITION.items() if nid in G.nodes}

    edge_colors = ['black' for _ in G.edges()]
    edge_widths = [max(1.2, 0.8 * data.get('weight', 1)) for _, _, data in G.edges(data=True)]

    buf = io.BytesIO()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=[attr['size'] for _, attr in G.nodes(data=True)],
        node_color=[attr['color'] for _, attr in G.nodes(data=True)],
        edge_color=edge_colors,
        width=edge_widths,
        font_size=12
    )
    ax.set_title(f"Trajectory Network (pitch speed): {traj_str}")
    # タイトルとグラフの重なりを防ぐためのマージン調整
    plt.margins(0.1)
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, pad_inches=0.2)
    plt.close('all')
    buf.seek(0)
    return buf

# ======================================================================
# 10) 追加ユーティリティ
#     group_extraction_with_coords と plot_seq_matplotlib2
# ======================================================================

def group_extraction_with_coords(data):
    """
    試合日、イニング、打席番号でグループ化して
    連続投球の zone シーケンスと plate_x plate_z の座標列を取り出す
    返り値
      df_return: trajectory と coords を持つ DataFrame
      pitch_types: 各打席の球種列のリスト
      pitch_speeds: 各打席の実測球速列のリスト
    """
    gb = data.groupby(['game_date', 'inning', 'at_bat_number'])

    trajects = []
    pitch_types = []
    pitch_speeds = []
    coords_list = []

    for _, g in gb:
        if len(g) == 1:
            continue
        else:
            s = ''.join(map(str, g['zone']))
            trajects.append(s)

            coords = [( -float(x) if pd.notna(x) else x, float(z) if pd.notna(z) else z ) for x, z in zip(g['plate_x'], g['plate_z'])]
            coords_list.append(coords)

            pitch_types.append(list(g['pitch_name']))
            pitch_speeds.append(list(g['effective_speed']))

    df_return = pd.DataFrame({
        'trajectory': trajects,
        'coords': coords_list
    })

    return df_return, pitch_types, pitch_speeds

def plot_seq_matplotlib2(
    df_vis, pos,
    xlim=None, ylim=None,
    strike_zone=(-0.83, 0.83, 1.5, 3.5),
    color_map=None,
    legend=True,
    ax=None,
    pad_frac=0.10,
    pad_x=None, pad_y=None,
    include_strike_zone_in_bounds=True,
    square_view=True,
    square_anchor="center",
    invert_x=False,
    show_axes=True,
    x_ticks=None, y_ticks=None
):
    if color_map is None:
        color_map = {
            'F': '#1f77b4',
            'C': '#2ca02c',
            'S': '#d62728',
            'D': '#9467bd',
            'O': '#7f7f7f'
        }

    if pos < 0 or pos >= len(df_vis):
        raise IndexError(f"pos={pos} は df_vis の範囲外です（0..{len(df_vis)-1}）。")

    for col in ['coords', 'PitchCategory', 'PitchSpeed']:
        if col not in df_vis.columns:
            raise KeyError(f"df_vis に必須列 {col} がありません。")

    row   = df_vis.iloc[pos]
    coords = row['coords']
    cats   = list(str(row['PitchCategory']))
    spds   = list(row['PitchSpeed'])

    n = min(len(coords or []), len(cats), len(spds))
    if n <= 1:
        raise ValueError(f"pos={pos} のシーケンスは投球数が {n}（2 未満）のため描画できません。")

    xs = [coords[i][0] for i in range(n)]
    ys = [coords[i][1] for i in range(n)]
    if invert_x:
        xs = [-float(x) for x in xs]
        # also update strike zone if it's in data coordinates
        sxmin, sxmax, symin, symax = strike_zone
        strike_zone = (-sxmax, -sxmin, symin, symax)
    colors = [color_map.get(cats[i], '#7f7f7f') for i in range(n)]

    spds = np.array(spds[:n], dtype=float)
    s_min, s_max = float(np.min(spds)), float(np.max(spds))
    if s_max - s_min == 0:
        sizes = np.full_like(spds, 400, dtype=float)
    else:
        sizes = 100 + 1500 * (spds - s_min) / (s_max - s_min)

    created_fig = False
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        created_fig = True

    sxmin, sxmax, symin, symax = strike_zone
    rect = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                         fill=False, color='gray', linewidth=2, alpha=0.6)
    ax.add_patch(rect)

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors='black')

    for i in range(n - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            continue
        ax.arrow(x0, y0, dx, dy,
                 head_width=0.10, head_length=0.25,
                 fc='black', ec='black', linewidth=1.0, alpha=0.9,
                 length_includes_head=True)

    if legend:
        legend_labels = {
            'F': 'Fastball', 'C': 'Curveball', 'S': 'Slider',
            'D': 'Offspeed', 'O': 'Others'
        }
        handles = [mpatches.Patch(color=color_map[k], label=v)
                   for k, v in legend_labels.items()]
        ax.legend(handles=handles, loc='upper right', title='Pitch Category')

    coords_series = df_vis['coords'].dropna()
    all_x = [c[0] for coords_list in coords_series for c in coords_list]
    all_y = [c[1] for coords_list in coords_series for c in coords_list]

    if include_strike_zone_in_bounds:
        all_x += [sxmin, sxmax]
        all_y += [symin, symax]

    if xlim is None and all_x:
        x_min, x_max = min(all_x), max(all_x)
        x_range = max(x_max - x_min, 1e-6)
        if pad_x is not None:
            pad_x_val = float(pad_x)
        else:
            pad_x_val = x_range * float(pad_frac)
        x_auto = (x_min - pad_x_val, x_max + pad_x_val)
    elif xlim is not None:
        x_auto = tuple(xlim)
    else:
        x_auto = (-2.5, 2.5)

    if ylim is None and all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_range = max(y_max - y_min, 1e-6)
        if pad_y is not None:
            pad_y_val = float(pad_y)
        else:
            pad_y_val = y_range * float(pad_frac)
        y_auto = (y_min - pad_y_val, y_max + pad_y_val)
    elif ylim is not None:
        y_auto = tuple(ylim)
    else:
        y_auto = (-1.0, 5.0)

    if square_view:
        x_center = (x_auto[0] + x_auto[1]) / 2.0
        y_center = (y_auto[0] + y_auto[1]) / 2.0
        x_len = x_auto[1] - x_auto[0]
        y_len = y_auto[1] - y_auto[0]
        side = max(x_len, y_len)

        if square_anchor == "center":
            half = side / 2.0
            x_auto = (x_center - half, x_center + half)
            y_auto = (y_center - half, y_center + half)
        else:
            # anchor options: left/right/top/bottom/center
            if square_anchor in ("left", "min"):
                x_auto = (x_auto[0], x_auto[0] + side)
            elif square_anchor in ("right", "max"):
                x_auto = (x_auto[1] - side, x_auto[1])
            else:
                # default: align to center horizontally
                half = side / 2.0
                x_auto = (x_center - half, x_center + half)

            if square_anchor in ("bottom",):
                y_auto = (y_auto[0], y_auto[0] + side)
            elif square_anchor in ("top",):
                y_auto = (y_auto[1] - side, y_auto[1])
            else:
                half = side / 2.0
                y_auto = (y_center - half, y_center + half)

    ax.set_xlim(*x_auto)
    ax.set_ylim(*y_auto)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Pitch Sequence for No.{pos}")
    ax.grid(True, linestyle='--', alpha=0.1)

    # Axis labels and ticks
    if show_axes:
        ax.set_xlabel('Plate X')
        ax.set_ylabel('Plate Z')
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

    if created_fig:
        logger.debug("Displaying matplotlib figure in interactive mode")
        plt.show()

    return ax

def update_elements(tapped_node: Optional[Dict[str, Any]],
                    elements: List[Dict[str, Any]],
                    selected_nodes: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Cytoscapeノードタップ時の要素更新処理。

    Parameters
    ----------
    tapped_node : dict | None
        タップされたノードのdata (dash_cytoscapeが渡す). Noneなら更新なし。
    elements : list[dict]
        現在のグラフ要素 (nodes + edges)。
    selected_nodes : list[str] | None
        既に選択されているノードIDのリスト。None/空なら新規開始。

    Returns
    -------
    (new_elements, selected_zones_text, selected_nodes_list)
        更新後の要素 / 表示用テキスト / 選択ノードリスト。
    """
    if selected_nodes is None:
        selected_nodes = []

    # タップ無しならそのまま返す
    if tapped_node is None or 'id' not in tapped_node:
        zones_text = 'Selected Zone : ' + ', '.join(selected_nodes)
        return elements, zones_text, selected_nodes

    node_id = tapped_node.get('id')
    if not isinstance(node_id, str):  # 念のための型チェック
        logger.warning("Tapped node id が文字列ではありません: %r", node_id)
        zones_text = 'Selected Zone : ' + ', '.join(selected_nodes)
        return elements, zones_text, selected_nodes

    # 連続10回同じノードをタップした場合は無視（単純な誤操作対策）
    if len(selected_nodes) >= 10 and all(n == node_id for n in selected_nodes[-10:]):
        zones_text = 'Selected Zone : ' + ', '.join(selected_nodes)
        logger.debug("同一ノード連続タップを無視: %s", node_id)
        return elements, zones_text, selected_nodes

    # ノード追加（履歴上限で古いものを削除）
    MAX_HISTORY = 50
    selected_nodes.append(node_id)
    if len(selected_nodes) > MAX_HISTORY:
        removed = selected_nodes.pop(0)
        logger.debug("選択履歴上限超過により削除: %s", removed)
    logger.debug("選択ノード更新: %s", selected_nodes)

    new_elements = list(elements)  # シャローコピー

    # 直近2ノードからエッジ生成（重複IDは生成しない）
    if len(selected_nodes) >= 2:
        source, target = selected_nodes[-2], selected_nodes[-1]
        edge_id = f"{source}-{target}"
        if not any(e.get('data', {}).get('id') == edge_id for e in new_elements if isinstance(e, dict)):
            new_elements.append({
                'data': {'source': source, 'target': target, 'id': edge_id},
                'classes': 'new-edge'
            })
            logger.debug("新規エッジ生成: %s -> %s", source, target)
        else:
            logger.debug("既存エッジのため生成スキップ: %s", edge_id)

    # ノード選択状態フラグを更新
    selected_set = set(selected_nodes)
    for node in new_elements:
        data = node.get('data') if isinstance(node, dict) else None
        if data and 'id' in data:
            data['selected'] = data['id'] in selected_set

    zones_text = 'Selected Zone : ' + ', '.join(selected_nodes)
    return new_elements, zones_text, selected_nodes

# ======================================================================
# 11) Run
# ======================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
