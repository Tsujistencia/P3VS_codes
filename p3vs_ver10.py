# ======================================================================
# 1) Imports and Configuration
# ======================================================================
import logging
import os
import json
import pickle
import base64
import io
from pathlib import Path
from hashlib import md5
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib
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

# Configure matplotlib to use non-GUI backend
matplotlib.use('Agg')

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
THUMBNAIL_HEIGHT = '180px'

# Clustering parameters - Distance cost weights
EDIT_DISTANCE_TRAJECTORY_COST = 2.83      # Insert/delete cost for trajectory matching
EDIT_DISTANCE_SPEED_COST = 1.0            # Insert/delete cost for speed sequences
EDIT_DISTANCE_PITCH_COST = 0.5            # Insert/delete cost for pitch type sequences

# Data settings
DEFAULT_SEASON = 2023
STANCE_TYPES = ['All', 'L', 'R']
MIN_CLUSTER_SIZE = 2
DEFAULT_RANDOM_STATE = 0
MAX_CLUSTER_IMAGES = 2000  # 最大表示画像数（パフォーマンス制限）
MAX_CLUSTERING_CACHE_SIZE = 100  # クラスタリングキャッシュの最大エントリ数

# Strike zone definition
STRIKE_ZONES = frozenset(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
BALL_ZONES = frozenset(['j', 'k', 'l', 'm'])
ALLOWED_ZONES = STRIKE_ZONES | BALL_ZONES

# Strike zone visualization parameters
STRIKE_ZONE_BOUNDS = (-0.83, 0.83, 1.5, 3.5)  # (x_min, x_max, z_min, z_max)
PLOT_PADDING_FRACTION = 0.10  # パディングの割合（bounds の何%）
PLOT_PAD_X = None  # X軸パディング（Noneの場合は PLOT_PADDING_FRACTION を使用）
PLOT_PAD_Y = None  # Y軸パディング（Noneの場合は PLOT_PADDING_FRACTION を使用）

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

SPEED_CATEGORIES = {
    'S': 0.6,
    'M': 1.0,
    'H': 1.4
}

# ======================================================================
# 4) Global Caches
# ======================================================================
NETWORK_IMAGE_CACHE = {}
CLUSTERING_CACHE = {}  # キャッシュ: (data_hash, num_classes, w_traj, w_speed, w_pitch) -> labels

# ======================================================================
# 0) Initialize Logger and Cache
# ======================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize imports
from config import (
    EVENT_MAP, CATEGORY_MAP, RESULT_EVENT_MAP,
    NODE_POSITION, SPEED_COORDINATES
)
from distance_utils import DistanceCalculator

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

def ensure_dirs(paths: list[Path]) -> None:
    """ディレクトリが存在することを確認し、必要に応じて作成する。
    
    Args:
        paths: Path オブジェクトのリスト
        
    Raises:
        OSError: ディレクトリ作成に失敗した場合
    """
    for p in paths:
        try:
            p.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ready: {p}")
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {p}: {e}")
            raise
        except FileExistsError as e:
            logger.warning(f"File exists where directory expected {p}: {e}")
            raise
        except OSError as e:
            logger.warning(f"mkdir failed for {p}, trying os.makedirs: {e}")
            try:
                import os as _os
                _os.makedirs(str(p), exist_ok=True)
                logger.debug(f"Directory created with os.makedirs: {p}")
            except OSError as os_error:
                logger.error(f"Failed to create directory {p} with both methods: {os_error}")
                raise

ensure_dirs([IMAGE_DIRECTORY, DATA_DIRECTORY_BASE, CACHE_DIRECTORY])

# ======================================================================
# 2) Initialize Distance Calculator
# ======================================================================

# ======================================================================
# 2) Helper Functions
# ======================================================================

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

def clustering_data(filtered_data: pd.DataFrame, num_classes: int) -> np.ndarray:

    if filtered_data.empty:
        return np.array([])

    num_classes = int(num_classes)
    trajectories = filtered_data['trajectory'].tolist()
    n = len(trajectories)

    # クラスター数の調整
    if n < num_classes:
        num_classes = n
    if num_classes == 0:
        return np.array([])

    # 距離行列の計算
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # 対称行列なので上三角部分のみ計算
            d = Levenshtein.distance(trajectories[i], trajectories[j])
            dist[i, j] = dist[j, i] = d

    # KMedoidsクラスタリングの実行
    try:
        kmedoids = KMedoids(
            n_clusters=num_classes,
            random_state=0,
            init='k-medoids++',
            metric='precomputed'
        )
        kmedoids.fit(dist)
        logger.info(f"KMedoids clustering completed: {num_classes} clusters from {n} samples")
        return kmedoids.labels_
    except ValueError as ve:
        logger.error(f"Invalid parameters for KMedoids: {ve}")
        return np.array([])
    except RuntimeError as re:
        logger.error(f"KMedoids fitting failed to converge: {re}")
        return np.array([])
    except Exception as e:
        logger.error(f"Unexpected error during KMedoids clustering: {type(e).__name__}: {e}", exc_info=True)
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

# ゾーンと許可ノードの定義（グローバル定数）
STRIKE_ZONES = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
BALL_ZONES = set(['j', 'k', 'l', 'm'])
ALLOWED_ZONES = STRIKE_ZONES | BALL_ZONES
ALLOWED_NODES = list('abcdefghijklm')
STRIKE_ZONE_NODES = list('abcdefghi')
BALL_ZONE_NODES = list('jklm')
IMAGE_EXTENSION = '.png'
CASH_DIRECTORY = os.path.join(HOME, 'Desktop', 'Cash')

def update_NX(pitcher_name: str, year: int, stance_type: str, selected_zones: str, 
              num_classes: int, categories: List[str]) -> List[html.Div]:
    """カテゴリごとのクラスター画像をHTML要素として返す。
    
    Args:
        pitcher_name: 投手の名前
        year: シーズン（年）
        stance_type: スタンスタイプ（All, L, R）
        selected_zones: 選択されたゾーン（カンマ区切り）
        num_classes: クラスター数
        categories: 投球カテゴリのリスト
        
    Returns:
        カテゴリごとの画像HTML要素のリスト
    """
    images_html = []
    
    for category in categories:  
        category_images = []  
        
        for i in range(int(num_classes)):
            filename = f"{pitcher_name}_{year}_{stance_type}_{category}_{selected_zones}_{num_classes}_{i}{IMAGE_EXTENSION}"
            image_path = os.path.join(IMAGE_DIRECTORY, filename)

            if os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        img = base64.b64encode(img_file.read()).decode('utf-8')
                        # Use fixed pixel thumbnail size for more consistent and larger display
                        category_images.append(html.Img(src=f'data:image/png;base64,{img}',
                                                        style={'height': THUMBNAIL_HEIGHT, 'width': THUMBNAIL_HEIGHT, 'object-fit': 'contain'}))
                except (IOError, OSError) as e:
                    logger.warning(f'Error reading image {image_path}: {e}')
                    category_images.append(html.Div(f"Error loading: {filename}"))
            else:
                logger.debug(f'Image file not found: {image_path}')
                category_images.append(html.Div(f"Image not found: {filename}"))

        category_html = html.Div([
            html.H4(f"Category: {category}"),
            html.Div(category_images, style={'display': 'flex', 'flex-direction': 'row', 'gap': '10px','justify-content':'center'})
        ])
        images_html.append(category_html)

    return images_html

# Initialize distance calculator
distance_calculator = DistanceCalculator(NODE_POSITION, SPEED_COORDINATES)

def weighted_edit_distance(s1: str, s2: str, cost_func: callable, ins_del_cost: float = 1.0) -> float:
    """加重編集距離を計算する。
    
    Args:
        s1: 最初の文字列
        s2: 2番目の文字列
        cost_func: 置換コストを計算するコールバック関数
        ins_del_cost: 挿入・削除の単位コスト
        
    Returns:
        加重編集距離
    """
    return distance_calculator.weighted_edit_distance(s1, s2, cost_func, ins_del_cost)

def get_trajectory_cost(c1: str, c2: str, node_pos: dict = NODE_POSITION) -> float:

    return distance_calculator.get_trajectory_cost(c1, c2)

def get_speed_cost(c1: str, c2: str, speed_pos: dict = SPEED_COORDINATES) -> float:

    return distance_calculator.get_speed_cost(c1, c2)

def get_pitch_type_cost(c1: str, c2: str, pitch_coords: dict) -> float:

    return distance_calculator.get_pitch_type_cost(c1, c2, pitch_coords)

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:

    return distance_calculator.normalize_matrix(matrix)

# ======================================================================
# 2.5) Utility Functions
# ======================================================================

def safe_file_read(filepath: Path, binary: bool = False) -> bytes | str | None:
    """ファイルを安全に読み込む。
    
    Args:
        filepath: ファイルパス
        binary: バイナリモードで読み込むか
        
    Returns:
        ファイル内容、またはエラー時はNone
    """
    try:
        mode = 'rb' if binary else 'r'
        with filepath.open(mode) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f'File not found: {filepath}')
        return None
    except IOError as e:
        logger.error(f'IO error reading {filepath}: {e}')
        return None
    except Exception as e:
        logger.error(f'Error reading {filepath}: {type(e).__name__}: {e}')
        return None

def safe_file_write(filepath: Path, content: bytes | str, binary: bool = False) -> bool:
    """ファイルを安全に書き込む。
    
    Args:
        filepath: ファイルパス
        content: 書き込み内容
        binary: バイナリモードで書き込むか
        
    Returns:
        成功時True、失敗時False
    """
    try:
        mode = 'wb' if binary else 'w'
        with filepath.open(mode) as f:
            f.write(content)
        logger.debug(f'Successfully wrote file: {filepath}')
        return True
    except OSError as e:
        logger.error(f'OS error writing {filepath}: {e}')
        return False
    except Exception as e:
        logger.error(f'Error writing {filepath}: {type(e).__name__}: {e}')
        return False

def validate_parameters(**kwargs) -> bool:
    """パラメータをバリデーションする。
    
    Args:
        **kwargs: {パラメータ名: (値, 期待される型)}
        
    Returns:
        すべてのパラメータが有効な場合True
    """
    for param_name, (value, expected_type) in kwargs.items():
        if not isinstance(value, expected_type):
            logger.warning(f'Parameter {param_name} is {type(value).__name__}, expected {expected_type.__name__}')
            return False
    return True


def performance_timer(func):
    """関数の実行時間をログに記録するデコレータ。
    
    Args:
        func: タイム計測対象の関数
        
    Returns:
        ラッパー関数
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f'{func.__name__} completed in {elapsed_time:.3f} seconds')
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f'{func.__name__} failed after {elapsed_time:.3f} seconds: {e}', exc_info=True)
            raise
    return wrapper


def get_clustering_cache_key(df: pd.DataFrame, num_classes: int, w_traj: float, 
                             w_speed: float, w_pitch: float) -> str:
    """クラスタリングキャッシュキーを生成する。
    
    DataFrameとパラメータのハッシュを基に一意なキーを生成する。
    
    Args:
        df: クラスタリング対象のデータフレーム
        num_classes: クラスター数
        w_traj: 軌跡距離の重み
        w_speed: 速度シーケンス距離の重み
        w_pitch: 投球タイプ距離の重み
        
    Returns:
        キャッシュキー（16文字のハッシュ）
    """
    try:
        # データフレームとパラメータをシリアライズ
        df_hash = md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        params_str = f"{num_classes}_{w_traj}_{w_speed}_{w_pitch}"
        params_hash = md5(params_str.encode('utf-8')).hexdigest()
        cache_key = f"{df_hash[:8]}_{params_hash[:8]}"
        logger.debug(f'Generated clustering cache key: {cache_key}')
        return cache_key
    except Exception as e:
        logger.error(f'Error generating clustering cache key: {e}')
        return None


def get_cached_clustering(df: pd.DataFrame, num_classes: int, w_traj: float,
                          w_speed: float, w_pitch: float) -> Optional[np.ndarray]:
    """キャッシュからクラスタリング結果を取得する。
    
    Args:
        df: クラスタリング対象のデータフレーム
        num_classes: クラスター数
        w_traj: 軌跡距離の重み
        w_speed: 速度シーケンス距離の重み
        w_pitch: 投球タイプ距離の重み
        
    Returns:
        キャッシュがあればラベル配列、なければNone
    """
    cache_key = get_clustering_cache_key(df, num_classes, w_traj, w_speed, w_pitch)
    if cache_key is None:
        return None
    
    if cache_key in CLUSTERING_CACHE:
        logger.info(f'Cache hit for clustering key: {cache_key}')
        return CLUSTERING_CACHE[cache_key]
    
    return None


def save_matplotlib_figure(filepath: Path, dpi: int = 100, bbox_inches: str = 'tight',
                          pad_inches: float = 0.0, format_type: str = 'png') -> bool:
    """matplotlib 図をファイルに保存し、リソースをクリーンアップする。
    
    Args:
        filepath: 保存先ファイルパス
        dpi: 解像度（dpi）
        bbox_inches: バウンディングボックス設定
        pad_inches: パディング
        format_type: ファイル形式（png, pdf など）
        
    Returns:
        成功時 True、失敗時 False
    """
    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, format=format_type)
        logger.debug(f'Successfully saved figure to {filepath}')
        return True
    except IOError as ioe:
        logger.error(f'IO error saving figure to {filepath}: {ioe}')
        return False
    except OSError as ose:
        logger.error(f'OS error saving figure to {filepath}: {ose}')
        return False
    except Exception as exc:
        logger.error(f'Unexpected error saving figure to {filepath}: {type(exc).__name__}: {exc}', exc_info=True)
        return False
    finally:
        try:
            plt.close('all')
        except Exception as e:
            logger.warning(f'Error closing matplotlib figures: {e}')


def save_matplotlib_buffer(buffer: io.BytesIO, dpi: int = 100, bbox_inches: str = 'tight',
                          pad_inches: float = 0.0, format_type: str = 'png') -> bool:
    """matplotlib 図をバイナリバッファに保存し、リソースをクリーンアップする。
    
    Args:
        buffer: 保存先バイオバッファ
        dpi: 解像度（dpi）
        bbox_inches: バウンディングボックス設定
        pad_inches: パディング
        format_type: ファイル形式（png, pdf など）
        
    Returns:
        成功時 True、失敗時 False
    """
    try:
        plt.savefig(buffer, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, format=format_type)
        logger.debug(f'Successfully saved figure to buffer (format={format_type})')
        return True
    except IOError as ioe:
        logger.error(f'IO error saving figure to buffer: {ioe}')
        return False
    except OSError as ose:
        logger.error(f'OS error saving figure to buffer: {ose}')
        return False
    except Exception as exc:
        logger.error(f'Unexpected error saving figure to buffer: {type(exc).__name__}: {exc}', exc_info=True)
        return False
    finally:
        try:
            plt.close('all')
        except Exception as e:
            logger.warning(f'Error closing matplotlib figures: {e}')


def filter_dataframe_by_category(df: pd.DataFrame, category: str, category_column: str = 'Category') -> pd.DataFrame:
    """指定されたカテゴリでデータフレームをフィルタリングし、インデックスをリセットする。
    
    Args:
        df: フィルタリング対象のデータフレーム
        category: フィルタリング対象のカテゴリ（'All'の場合は全データを返す）
        category_column: カテゴリ列の名前
        
    Returns:
        フィルタリング済みのデータフレーム（インデックスはリセット）
    """
    if category == 'All':
        return df.reset_index(drop=True)
    else:
        return df[df[category_column] == category].reset_index(drop=True)


def safe_get_dataframe_value(df: pd.DataFrame, row_idx: int, column: str, default: Any = None) -> Any:
    """データフレームから安全に値を取得する。
    
    Args:
        df: データフレーム
        row_idx: 行インデックス
        column: 列名
        default: デフォルト値（取得失敗時）
        
    Returns:
        指定された値、または default
    """
    try:
        if 0 <= row_idx < len(df) and column in df.columns:
            value = df.iloc[row_idx][column]
            return value if pd.notna(value) else default
        return default
    except (IndexError, KeyError) as e:
        logger.debug(f'Error accessing DataFrame[{row_idx}, {column}]: {e}')
        return default
    except Exception as e:
        logger.warning(f'Unexpected error accessing DataFrame value: {e}')
        return default


def cache_clustering_result(df: pd.DataFrame, num_classes: int, w_traj: float,
                           w_speed: float, w_pitch: float, labels: np.ndarray) -> None:
    """クラスタリング結果をキャッシュに保存する。
    
    Args:
        df: クラスタリング対象のデータフレーム
        num_classes: クラスター数
        w_traj: 軌跡距離の重み
        w_speed: 速度シーケンス距離の重み
        w_pitch: 投球タイプ距離の重み
        labels: クラスターラベル配列
    """
    cache_key = get_clustering_cache_key(df, num_classes, w_traj, w_speed, w_pitch)
    if cache_key is not None:
        CLUSTERING_CACHE[cache_key] = labels
        logger.debug(f'Cached clustering result for key: {cache_key}')
        # キャッシュサイズを制限
        if len(CLUSTERING_CACHE) > MAX_CLUSTERING_CACHE_SIZE:
            oldest_key = next(iter(CLUSTERING_CACHE))
            del CLUSTERING_CACHE[oldest_key]
            logger.debug(f'Removed oldest clustering cache entry: {oldest_key}')

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
            dcc.RadioItems(id='numc', options=[{'label': str(i), 'value': str(i)} for i in range(1, 7)],
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
@performance_timer
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
    path = DATA_DIRECTORY_BASE / f'{name}.csv'
    if not path.exists():
        logger.warning(f'Data file not found for pitcher {name} at {path}')
        return pd.DataFrame(), {}
    
    try:
        data = pd.read_csv(path)
        logger.info(f'Successfully loaded data for {name}: {len(data)} rows')
    except FileNotFoundError as fnf:
        logger.error(f'File not found even after existence check {path}: {fnf}')
        return pd.DataFrame(), {}
    except pd.errors.ParserError as pe:
        logger.error(f'CSV parsing error in {path}: {pe}')
        return pd.DataFrame(), {}
    except pd.errors.EmptyDataError as ede:
        logger.error(f'CSV file is empty {path}: {ede}')
        return pd.DataFrame(), {}
    except Exception as exc:
        logger.error(f'Unexpected error reading CSV {path}: {type(exc).__name__}: {exc}', exc_info=True)
        return pd.DataFrame(), {}

    # 日付とシーズンの処理
    try:
        data['game_date'] = pd.to_datetime(data['game_date'])
        data['year'] = data['game_date'].dt.year
    except Exception as e:
        logger.error(f'Error processing date columns: {type(e).__name__}: {e}')
        return pd.DataFrame(), {}
    
    # シーズンリストのバリデーション
    if not isinstance(season_list, list) or not season_list:
        season_list = [2024]
        logger.warning('Invalid season_list provided, using default [2024]')
    
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

@performance_timer
def perform_clustering(df: pd.DataFrame, num_classes_str: str, w_traj: float, 
                       w_speed: float, w_pitch: float, pitch_coords: dict) -> np.ndarray:
    """複数のメトリクスを組み合わせてクラスタリングを実行する。
    
    Args:
        df: 軌跡、速度情報を含むデータフレーム
        num_classes_str: クラスター数（文字列形式）
        w_traj: 軌跡距離の重み
        w_speed: 速度シーケンス距離の重み
        w_pitch: 投球タイプ距離の重み
        pitch_coords: 投球タイプの座標情報
        
    Returns:
        np.ndarray: クラスターラベルの配列
    """
    trajectories, n, num_classes = df.to_dict('records'), len(df), int(num_classes_str)
    
    if n < num_classes:
        num_classes = n
        logger.debug(f'Adjusted num_classes from {num_classes_str} to {num_classes}')
    if num_classes == 0:
        logger.warning('num_classes is 0, returning empty array')
        return np.array([])
    
    logger.info(f'Starting clustering: {n} samples, {num_classes} clusters')
    
    dist_traj, dist_speed, dist_pitch = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist_traj[i, j] = dist_traj[j, i] = weighted_edit_distance(
                trajectories[i]['trajectory'], trajectories[j]['trajectory'],
                get_trajectory_cost, ins_del_cost=EDIT_DISTANCE_TRAJECTORY_COST
            )
            dist_speed[i, j] = dist_speed[j, i] = weighted_edit_distance(
                trajectories[i]['SpeedSequence'], trajectories[j]['SpeedSequence'],
                get_speed_cost, ins_del_cost=EDIT_DISTANCE_SPEED_COST
            )
            cost_func = lambda c1, c2: get_pitch_type_cost(c1, c2, pitch_coords)
            dist_pitch[i, j] = dist_pitch[j, i] = weighted_edit_distance(
                trajectories[i]['PitchSequence'], trajectories[j]['PitchSequence'],
                cost_func, ins_del_cost=EDIT_DISTANCE_PITCH_COST
            )
    
    final_dist = (w_traj * normalize_matrix(dist_traj)
                  + w_speed * normalize_matrix(dist_speed)
                  + w_pitch * normalize_matrix(dist_pitch))
    
    logger.debug(f'Distance matrix computed: shape={final_dist.shape}')
    
    kmedoids = KMedoids(n_clusters=num_classes, random_state=0, metric='precomputed', init='k-medoids++')
    try:
        kmedoids.fit(final_dist)
        logger.info(f'KMedoids clustering completed successfully')
        return kmedoids.labels_
    except ValueError as ve:
        logger.error(f'Invalid parameters for KMedoids: {ve}')
        return np.array([])
    except RuntimeError as re:
        logger.error(f'KMedoids fitting failed to converge: {re}')
        return np.array([])
    except Exception as e:
        logger.error(f'Unexpected error during KMedoids clustering: {type(e).__name__}: {e}', exc_info=True)
        return np.array([])

# --- helpers for cluster images ---
def class_choice(clustered_data: pd.DataFrame, class_label: int) -> pd.Series:

    return clustered_data[clustered_data['class'] == class_label]['trajectory']

def nodeCount0(trajectories: list[str]) -> Counter:

    return Counter(char for traj in trajectories for char in traj)

def edgeCount(trajectories: list[str]) -> Counter:

    return Counter(edge for traj in trajectories if len(traj) > 1 for edge in zip(traj[:-1], traj[1:]))

def edge_merge(sorted_edges_with_counts: list[tuple]) -> dict:

    merged = {}
    for edge, count in sorted_edges_with_counts:
        canonical = tuple(sorted(edge))
        merged[canonical] = merged.get(canonical, 0) + count
    return merged

# ======================================================================
# 4.5) Cluster Overview Network
@performance_timer
def create_NX(clustered_data: pd.DataFrame, pitcher_name: str, year_str: str, 
              stance_type: str, seq_str: str, num_classes: int, category: str, 
              nodesize_unit: float, min_edge_weight: float) -> None:
    """クラスタネットワークの概観図を作成して PNG ファイルとして保存する。
    
    Args:
        clustered_data: クラスタリング済みデータフレーム
        pitcher_name: 投手の名前
        year_str: 年度（文字列）
        stance_type: 打者のスタンス（L/R）
        seq_str: シーケンスフィルター文字列
        num_classes: クラスター数
        category: カテゴリー
        nodesize_unit: ノードサイズの基本単位
        min_edge_weight: エッジ表示の最小重み
        
    Returns:
        None
    """
    logger.info(f"create_NX called with min_edge_weight={min_edge_weight}")
    
    # クリーンアップ：この投手/年/スタンス/カテゴリの古い画像を削除
    logger.info(f"Cleaning up old images for pattern: {pitcher_name}_{year_str}_{stance_type}_{category}_*_*.png")
    try:
        for old_file in IMAGE_DIRECTORY.glob(f'{pitcher_name}_{year_str}_{stance_type}_{category}_*_*.png'):
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
        # ノードの色を濃くする（alpha=1.0で完全不透明）
        G.add_nodes_from(
            (n, {"color": "#FF8C00" if n in STRIKE_ZONE_NODES else "#1E90FF"})  # darkorange → より濃いオレンジ, skyblue → より濃い青
            for n in ALLOWED_NODES
        )

        nodesize = [
            nodesize_unit * node00.get(n, 0) if node00.get(n, 0) > 0 else nodesize_unit / 2
            for n in G.nodes
        ]

        for (u, v), cnt in edge_data00.items():
            if cnt < min_edge_weight or u == v:
                continue
            if (u not in ALLOWED_NODES) or (v not in ALLOWED_NODES):
                continue

            if (u in BALL_ZONE_NODES) and (v in BALL_ZONE_NODES):
                base_color = '#1f6fe5'
            elif (u in STRIKE_ZONE_NODES) and (v in STRIKE_ZONE_NODES):
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
            # エッジの透明度を頻度に応じて調整: 低頻度=薄く(0.2), 高頻度=濃く(1.0)
            alpha = 0.2 + 0.8 * (w / max_w) if max_w > 0 else 1.0
            edge_colors_rgba.append(to_rgba(base_color, alpha))
            edge_widths.append(max(1.5, 0.5 * w))

        filename = f'{pitcher_name}_{year_str}_{stance_type}_{category}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
        update_Path = IMAGE_DIRECTORY / filename
        logger.info(f"Generating image: {filename} with min_edge_weight={min_edge_weight}")
        if update_Path.exists():
            logger.info('File already exists, skipping: %s', filename)
            continue

        plt.figure(figsize=(8, 8))
        # ノードの色を濃く表示（alpha=1.0で完全不透明）
        node_colors_rgba = [to_rgba(n_attr["color"], 1.0) for _, n_attr in G.nodes(data=True)]
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=nodesize,
            node_color=node_colors_rgba,
            edgelist=[(u, v) for u, v, _ in edge_list_ordered],
            edge_color=edge_colors_rgba,
            width=edge_widths,
            font_size=20
        )
        plt.title(f"Cluster {i+1} for {category} ({len(test_data)} sequences)")
        save_matplotlib_figure(update_Path, dpi=100)

# ======================================================================
# 5) Helper UIs
# ======================================================================

def update_image_display(pitcher_name: str, seasons: list, stance_type: str, 
                         selected_zones: list, num_classes: int, 
                         categories: list, min_edge_weight: float) -> html.Div:
    year_str = "_".join(map(str, sorted(seasons)))
    seq_str = "".join(selected_zones) if selected_zones else "any"
    categories_to_process = ['All'] if 'All' in categories else categories

    category_blocks = []
    for cat in categories_to_process:
        cat_images = []
        for i in range(int(num_classes)):
            filename = f'{pitcher_name}_{year_str}_{stance_type}_{cat}_{seq_str}_{num_classes}_{min_edge_weight}_{i}.png'
            image_path = IMAGE_DIRECTORY / filename
            if image_path.exists():
                try:
                    with image_path.open('rb') as fh:
                        img_bytes = fh.read()
                    if not img_bytes:
                        logger.warning(f'Image file is empty: {image_path}')
                        continue
                    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
                    cat_images.append(html.Img(src=f'data:image/png;base64,{encoded_img}', style={'height': '300px', 'margin': '5px'}))
                    logger.debug(f'Successfully encoded image: {image_path}')
                except FileNotFoundError as fnf:
                    logger.warning(f'Image file not found: {image_path}: {fnf}')
                except IOError as ioe:
                    logger.error(f'Error reading image file {image_path}: {ioe}')
                except (ValueError, UnicodeDecodeError) as dce:
                    logger.error(f'Error encoding image {image_path}: {dce}')
                except Exception as exc:
                    logger.error(f'Unexpected error processing image {image_path}: {type(exc).__name__}: {exc}', exc_info=True)
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

def draw_single_sequence_image(traj_str: str) -> tuple[html.Img | html.Div, str | None]:
    if not traj_str or len(traj_str) < 2:
        return html.Div("Select one trajectory row."), None
    key = md5(traj_str.encode('utf-8')).hexdigest()[:12]
    fname = f"seq_{key}.png"
    fpath = IMAGE_DIRECTORY / fname
    if not fpath.exists():
        pos = {k: ((v[0]+3)*1.0, (v[1]+3)*1.0) for k, v in NODE_POSITION.items()}
        plt.figure(figsize=(6, 6))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        for nid, (x, y) in pos.items():
            color = 'orange' if nid in list('abcdefghi') else 'lightblue'
            # use RGBA with alpha for node markers
            ax.scatter(x, y, s=300, c=[to_rgba(color, 0.7)], edgecolors='black')
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
        save_matplotlib_figure(fpath, dpi=150, pad_inches=0.2)
    try:
        with fpath.open('rb') as fh:
            encoded = base64.b64encode(fh.read()).decode()
    except Exception:
        logger.exception('Failed to read generated sequence image: %s', fpath)
        return html.Div('Could not open generated image.'), None
    return html.Img(src=f"data:image/png;base64,{encoded}", style={'height':'360px','margin':'6px'}), str(fpath)

# ======================================================================
# 5) Analysis Helper Functions (Refactored)
# ======================================================================

def load_cluster_payload_cache(p_name: str, year_str: str, s_type: str, 
                                categories: List[str], seq_str: str, num_c: int,
                                w_t: float, w_s: float, w_p: float) -> Tuple[Dict, Dict, bool]:
    """クラスタペイロードキャッシュをロード
    
    Args:
        p_name: 投手名
        year_str: 年度文字列  
        s_type: スタンスタイプ
        categories: カテゴリリスト
        seq_str: シーケンス文字列
        num_c: クラスタ数
        w_t, w_s, w_p: 重み係数
        
    Returns:
        (cluster_payload, cluster_bounds, cache_loaded): キャッシュデータとロード成功フラグ
    """
    payload_cache_params = f"{p_name}_{year_str}_{s_type}_{'_'.join(categories)}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
    payload_cache_filename = f"cluster_payload_{payload_cache_params}.pkl"
    payload_cache_filepath = CACHE_DIRECTORY / payload_cache_filename
    
    if not payload_cache_filepath.exists():
        return {}, {}, False
    
    try:
        with payload_cache_filepath.open('rb') as f:
            cached_data = pickle.load(f)
        cluster_payload = cached_data.get('payload', {})
        cluster_bounds = cached_data.get('bounds', {})
        logger.info(f'Loaded cluster payload from cache: {payload_cache_filename} ({len(cluster_payload)} clusters)')
        return cluster_payload, cluster_bounds, True
    except Exception:
        logger.exception(f'Failed loading cluster payload cache {payload_cache_filepath}')
        return {}, {}, False


def save_cluster_payload_cache(p_name: str, year_str: str, s_type: str,
                                categories: List[str], seq_str: str, num_c: int,
                                w_t: float, w_s: float, w_p: float,
                                cluster_payload: Dict, cluster_bounds: Dict) -> bool:
    """クラスタペイロードキャッシュを保存
    
    Args:
        p_name: 投手名
        year_str: 年度文字列
        s_type: スタンスタイプ
        categories: カテゴリリスト
        seq_str: シーケンス文字列
        num_c: クラスタ数
        w_t, w_s, w_p: 重み係数
        cluster_payload: クラスタペイロードデータ
        cluster_bounds: クラスタ境界データ
        
    Returns:
        bool: 保存成功フラグ
    """
    payload_cache_params = f"{p_name}_{year_str}_{s_type}_{'_'.join(categories)}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
    payload_cache_filename = f"cluster_payload_{payload_cache_params}.pkl"
    payload_cache_filepath = CACHE_DIRECTORY / payload_cache_filename
    
    try:
        with payload_cache_filepath.open('wb') as f:
            pickle.dump({'payload': cluster_payload, 'bounds': cluster_bounds}, f)
        logger.info(f'Saved cluster payload with thumbnails to cache: {payload_cache_filename}')
        return True
    except Exception:
        logger.exception(f'Failed saving cluster payload cache {payload_cache_filepath}')
        return False


def compute_cluster_axis_limits(records: List[Dict]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """クラスタ全体の軸範囲を計算（外れ値除外）
    
    Args:
        records: クラスタのレコードリスト
        
    Returns:
        (xlim, ylim): X軸とY軸の範囲タプル、データがない場合はNone
    """
    all_xs, all_ys = [], []
    
    for rec in records:
        coords = rec.get('coords')
        if not coords:
            continue
            
        for c in coords:
            if not c or len(c) < 2:
                continue
            try:
                x, y = float(c[0]), float(c[1])
                # 基本的な妥当性チェック（外れ値除外）
                if -10 <= x <= 10 and -5 <= y <= 10:
                    all_xs.append(x)
                    all_ys.append(y)
            except (ValueError, TypeError):
                continue
    
    if not all_xs or not all_ys:
        return None, None
    
    # パーセンタイルで極端な外れ値を除外
    xs_array = np.array(all_xs)
    ys_array = np.array(all_ys)
    x_min, x_max = np.percentile(xs_array, [5, 95])
    y_min, y_max = np.percentile(ys_array, [5, 95])
    
    # 範囲に余裕を持たせる（20%パディング）
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    xlim = (x_min - 0.2 * x_range, x_max + 0.2 * x_range)
    ylim = (y_min - 0.2 * y_range, y_max + 0.2 * y_range)
    
    return xlim, ylim


def generate_cluster_thumbnails(records: List[Dict], xlim: Optional[Tuple[float, float]], 
                                 ylim: Optional[Tuple[float, float]]) -> List[Dict]:
    """クラスタのサムネイル画像を事前生成してbase64エンコード
    
    Args:
        records: クラスタのレコードリスト
        xlim: X軸範囲
        ylim: Y軸範囲
        
    Returns:
        List[Dict]: サムネイル付きレコードリスト
    """
    for rec in records:
        try:
            buf = generate_pitchspeed_network_png(
                rec.get('trajectory', ''),
                rec.get('PitchSequence', ''),
                rec.get('SpeedSequence', ''),
                coords_override=rec.get('coords'),
                xlim=xlim,
                ylim=ylim
            )
            if buf:
                rec['thumbnail_b64'] = base64.b64encode(buf.getvalue()).decode()
            else:
                rec['thumbnail_b64'] = None
        except Exception as e:
            logger.warning(f"Failed to pre-generate thumbnail for {rec.get('trajectory', '')}: {e}")
            rec['thumbnail_b64'] = None
    
    return records


# ======================================================================
# 6) Callbacks
# ======================================================================

@app.callback(
    Output('selected-zones-text', 'children'),
    Input('selected-nodes-store', 'data'),
    prevent_initial_call=False
)
def update_selected_zones_text(selected_nodes: Optional[List[str]]) -> str:
    """選択されたゾーンのテキスト表示を更新
    
    Args:
        selected_nodes: 選択されたノードのリスト
        
    Returns:
        str: 表示用テキスト
    """
    if not selected_nodes:
        return 'Selected Zone：'
    return f'Selected Zone：{", ".join(selected_nodes)}'


@app.callback(
    [Output('selected-nodes-store', 'data', allow_duplicate=True),
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
    ],
    prevent_initial_call=True
)
def handle_buttons(start_clicks: int, reset_clicks: int, tapped_node: Optional[Dict[str, Any]], 
                   pitcher_name: str, year: int, stance_type: str, num_classes: int, 
                   Category: str, selected_nodes: Optional[List[str]], elements: List[Dict]) -> Tuple[Optional[List[str]], List[Dict]]:
    """Dash callback: ボタン入力とノードタップを処理する。
    
    Args:
        start_clicks: Startボタンのクリック数
        reset_clicks: Resetボタンのクリック数
        tapped_node: Cytoscape内でタップされたノード情報
        pitcher_name: 投手の名前
        year: シーズン（年）
        stance_type: スタンスタイプ
        num_classes: クラスター数
        Category: 投球カテゴリ
        selected_nodes: 選択されたノードのリスト
        elements: Cytoscape要素
        
    Returns:
        (選択されたノードのリスト, 更新されたCytoscape要素)
    """
    nodesize_unit = 50
    ctx = dash.callback_context

    output_nodes = selected_nodes
    output_elements = elements

    if not ctx.triggered:
        return [], elements_init

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button' and reset_clicks > 0:
        # clear persisted selection on the helper function if present
        if hasattr(update_elements, 'selected_nodes'):
            delattr(update_elements, 'selected_nodes')
        return [], elements_init

    if button_id == 'dash_cyto_layout' and tapped_node is not None:
        # Delegate tap handling to update_elements helper and get updated elements/store
        output_elements, _, output_nodes = update_elements(
            tapped_node, elements=elements, selected_nodes=selected_nodes
        )

    return output_nodes, output_elements

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

    # Try to load cluster_payload from cache first (includes pre-generated thumbnails)
    cluster_payload, cluster_bounds, cache_loaded = load_cluster_payload_cache(
        p_name, year_str, s_type, categories_to_process, seq_str, num_c, w_t, w_s, w_p
    )
    
    # Process each category (either from cache or by clustering)
    for cat in categories_to_process:
        # Skip clustering if we have cached payload, but still need to generate network viz
        if cache_loaded and cluster_payload:
            # Load clustering result to generate network diagram
            cache_params = f"{p_name}_{year_str}_{s_type}_{cat}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
            cache_filename = f"clustered_{cache_params}.pkl"
            cache_filepath = CACHE_DIRECTORY / cache_filename
            
            if cache_filepath.exists():
                try:
                    with cache_filepath.open('rb') as f:
                        cat_df_with_labels = pickle.load(f)
                    # Restore coords if missing
                    if 'coords' not in getattr(cat_df_with_labels, 'columns', []):
                        traj_to_coords = dict(zip(full_df['trajectory'], full_df.get('coords', [])))
                        cat_df_with_labels = cat_df_with_labels.copy()
                        cat_df_with_labels['coords'] = cat_df_with_labels['trajectory'].map(traj_to_coords)
                    
                    # Generate network diagram from cached clustering result
                    try:
                        create_NX(cat_df_with_labels, p_name, year_str, s_type, seq_str, num_c, cat, node_size, min_edge)
                    except Exception:
                        logger.exception('create_NX failed for category %s', cat)
                except Exception:
                    logger.exception('Failed loading clustered cache for network viz %s', cache_filepath)
            continue
        
        # Normal clustering path (no cache)
        cache_params = f"{p_name}_{year_str}_{s_type}_{cat}_{seq_str}_{num_c}_{w_t}_{w_s}_{w_p}"
        cache_filename = f"clustered_{cache_params}.pkl"
        cache_filepath = CACHE_DIRECTORY / cache_filename

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
            cat_df = filter_dataframe_by_category(filtered, cat, 'Category')
            if cat_df.empty:
                continue

            # キャッシュをチェック
            cached_labels = get_cached_clustering(cat_df, int(num_c), w_t, w_s, w_p)
            if cached_labels is not None:
                labels = cached_labels
                logger.info(f'Using cached clustering result for category {cat}')
            else:
                labels = perform_clustering(cat_df, num_c, w_t, w_s, w_p, new_pitch_coords)
                if labels.size > 0:
                    cache_clustering_result(cat_df, int(num_c), w_t, w_s, w_p, labels)
            
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

        # Process each cluster
        for i in range(int(num_c)):
                key = f"{cat}_{i}"
                subset = cat_df_with_labels[cat_df_with_labels['class'] == i]
                if subset.empty:
                    continue
                records = subset[['trajectory', 'coords', 'PitchSequence', 'SpeedSequence']].to_dict('records')
                
                # Compute cluster-wide axis limits and generate thumbnails
                cluster_xlim, cluster_ylim = compute_cluster_axis_limits(records)
                records = generate_cluster_thumbnails(records, cluster_xlim, cluster_ylim)
                
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
    
    # Save cluster_payload with thumbnails to cache for fast loading (only if not loaded from cache)
    if not cache_loaded:
        save_cluster_payload_cache(
            p_name, year_str, s_type, categories_to_process, seq_str, num_c,
            w_t, w_s, w_p, cluster_payload, cluster_bounds
        )
    
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
def show_cluster_table(n_clicks_list: List[Optional[int]], cluster_store: Dict[str, List[Dict[str, Any]]]) -> Union[html.Div, str]:
    """クラスターシーケンスをグリッドで表示する Dash callback。
    
    このコールバックは、最大60個の小さなネットワーク画像を表示して、
    各軌跡がクラスター内でどのように配置されているかを示す。
    
    Args:
        n_clicks_list: クラスターボタンのクリック数リスト
        cluster_store: {キー: レコードのリスト} 形式のクラスターデータストア
        
    Returns:
        html.Div: 画像グリッドレイアウト、または状態テキスト
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

    # Limit images for performance
    images = []
    count = 0
    
    for rec in records:
        # if count >= MAX_CLUSTER_IMAGES:
        #     break
        
        traj = rec.get('trajectory', '')
        pitch_seq = rec.get('PitchSequence', '')
        speed_seq = rec.get('SpeedSequence', '')
        
        if not traj:
            continue
        
        try:
            # Use pre-generated thumbnail from cache if available
            encoded = rec.get('thumbnail_b64')
            
            if encoded is None:
                # Fallback to simple trajectory visualization if cache is not available
                img_comp, _ = draw_single_sequence_image(traj)
                images.append(html.Div(
                    [img_comp, html.Div(traj, style={'fontSize': '11px', 'fontWeight': 'bold', 'textAlign': 'center'})],
                    style={'textAlign': 'center', 'margin': '4px', 'border': '1px solid #ddd', 'padding': '4px', 'borderRadius': '4px'}
                ))
            else:
                # Use cached base64-encoded image (no re-generation needed!)
                img = html.Img(
                    src=f"data:image/png;base64,{encoded}",
                    style={'height': THUMBNAIL_HEIGHT, 'margin': '6px', 'border': '1px solid #ccc', 'borderRadius': '3px'}
                )
                caption = html.Div([
                    html.Div(traj, style={'fontSize': '10px', 'fontWeight': 'bold'}),
                    html.Div(f"P:{pitch_seq[:8]}... S:{speed_seq[:8]}...", style={'fontSize': '8px', 'color': '#666'})
                ], style={'textAlign': 'center', 'fontSize': '9px'})
                images.append(html.Div([img, caption], style={'textAlign': 'center', 'margin': '4px'}))
        except Exception as exc:
            logger.exception('Failed displaying image for trajectory %s: %s', traj, exc)
            continue

        count += 1
    
    # Create grid layout with flex wrapping
    grid = html.Div(
        images,
        style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'flex-start',
            'gap': '8px'
        }
    )
    
    # Header with summary info
    header = html.Div([
        html.H4(f"{cat} - Cluster {idx+1}", style={'marginBottom': '8px'}),
        html.P(f"Total sequences: {len(records)}  |  Images shown: {min(len(records), MAX_CLUSTER_IMAGES)}", 
               style={'fontSize': '12px', 'color': '#666'})
    ])
    
    return html.Div([header, grid], style={'padding': '10px'})

@app.callback(
    Output('seq-preview', 'children'),
    Output('network-url', 'data'),
    Input({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'active_cell'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'data'),
    State({'type':'seqfreq-table','category':ALL,'cluster':ALL}, 'id'),
    State('cluster-store', 'data'),
    State('cluster-bounds-store', 'data')
)
def preview_single_sequence_from_freq(
    active_cells, all_tables_data, all_tables_ids, cluster_store, cluster_bounds_store,
    *,
    # 追加オプション
    page_size=None,          # None なら全点表示。整数ならページ分割
    page=0,                  # 0 始まりのページ番号
    strike_zone=(-0.83, 0.83, 1.5, 3.5),
    include_strike_zone_in_bounds=True,
    pad_frac=0.10,
    pad_x=None, pad_y=None,
    square_view=True,
    square_anchor="center",
    invert_x=False,
    show_axes=True,
    x_ticks=None, y_ticks=None
):
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
    coords_full = rep.get('coords', [])  # plate_x, plate_z のリスト
    pitch_seq = rep.get('PitchSequence', '')
    speed_seq = rep.get('SpeedSequence', '')

    if not coords_full or len(coords_full) < 2:
        return html.Div(f"シーケンス {traj} は描画に必要な座標データがありません。"), no_update

    # 球種と速度の正規化
    if not pitch_seq:
        pitch_seq = 'F' * len(coords_full)
    if not speed_seq:
        speed_seq = [90.0 + i for i in range(len(coords_full))]
    else:
        try:
            speed_seq = [float(s) for s in speed_seq]
        except (ValueError, TypeError):
            speed_seq = [90.0 + i for i in range(len(coords_full))]

    # ページ分割処理
    total_n = len(coords_full)
    if page_size is None:
        start_idx, end_idx = 0, total_n
    else:
        page_size = max(1, int(page_size))
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_n)
        if start_idx >= total_n:
            # ページが範囲外なら最後のページに合わせる
            last_page = max(0, (total_n - 1) // page_size)
            start_idx = last_page * page_size
            end_idx = total_n
            page = last_page

    coords = coords_full[start_idx:end_idx]
    pitch_seq_seg = str(pitch_seq)[start_idx:end_idx]
    speed_seq_seg = list(speed_seq)[start_idx:end_idx]

    # 可視化用 DataFrame
    df_vis = pd.DataFrame({
        'coords': [coords],
        'PitchCategory': [pitch_seq_seg],
        'PitchSpeed': [speed_seq_seg]
    })

    # 軸範囲の決定
    bounds = cluster_bounds_store.get(key) if cluster_bounds_store else None
    xlim = ylim = None

    if bounds:
        # 事前計算済みのクラスタ境界があればそれを使う
        xlen = abs(bounds['xmax'] - bounds['xmin'])
        ylen = abs(bounds['ymax'] - bounds['ymin'])
        px = xlen * 0.10 if pad_x is None else float(pad_x)
        py = ylen * 0.10 if pad_y is None else float(pad_y)
        xlim = (bounds['xmin'] - px, bounds['xmax'] + px)
        ylim = (bounds['ymin'] - py, bounds['ymax'] + py)
    else:
        # 同クラスタの全 records から統一範囲を自動計算
        xs_all, ys_all = [], []
        for r in records:
            cs = r.get('coords') or []
            for x, y in cs:
                xs_all.append(float(-x) if invert_x else float(x))
                ys_all.append(float(y))
        if xs_all and ys_all:
            x_min, x_max = min(xs_all), max(xs_all)
            y_min, y_max = min(ys_all), max(ys_all)
            x_range = max(x_max - x_min, 1e-6)
            y_range = max(y_max - y_min, 1e-6)
            px = x_range * float(pad_frac) if pad_x is None else float(pad_x)
            py = y_range * float(pad_frac) if pad_y is None else float(pad_y)

            sxmin, sxmax, symin, symax = strike_zone
            if invert_x:
                sxmin, sxmax = -sxmax, -sxmin

            if include_strike_zone_in_bounds:
                x_min = min(x_min, sxmin); x_max = max(x_max, sxmax)
                y_min = min(y_min, symin); y_max = max(y_max, symax)

            xlim = (x_min - px, x_max + px) + 4
            ylim = (y_min - py, y_max + py) + 4
        # xs_all が無ければ plot_seq_matplotlib2 の自動に任せる

    # 描画
    try:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        plot_seq_matplotlib2(
            df_vis, pos=0, ax=ax,
            xlim=xlim, ylim=ylim,
            strike_zone=strike_zone,
            legend=True,
            pad_frac=pad_frac, pad_x=pad_x, pad_y=pad_y,
            include_strike_zone_in_bounds=include_strike_zone_in_bounds,
            square_view=square_view, square_anchor=square_anchor,
            invert_x=invert_x, show_axes=show_axes,
            x_ticks=x_ticks, y_ticks=y_ticks
        )

        # 画像化
        buf = io.BytesIO()
        save_matplotlib_buffer(buf, dpi=150, pad_inches=0.2)
        buf.seek(0)

        cache_key_src = f"{traj}|{pitch_seq_seg}|{speed_seq_seg}|{xlim}|{ylim}"
        cache_key = md5(cache_key_src.encode('utf-8')).hexdigest()[:16]
        NETWORK_IMAGE_CACHE[cache_key] = buf

        url = f"/dynamic_network/{cache_key}"

        # ページ表示用の付記
        page_info = ""
        if page_size is not None:
            total_pages = max(1, (total_n + page_size - 1) // page_size)
            page_info = f"  Page {page+1}/{total_pages}  [{start_idx+1}..{end_idx}] of {total_n}"

        return html.Div([
            html.H4(f"投球シーケンス: {traj.upper()}{page_info}"),
            html.P(f"球種: {pitch_seq_seg}"),
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
def serve_dynamic_network(key: str) -> Response:
    """メモリ内キャッシュからネットワーク画像を提供する。
    
    Args:
        key: ネットワーク画像のキャッシュキー
        
    Returns:
        Response: PNG画像データまたは404エラー
    """
    buf = NETWORK_IMAGE_CACHE.get(key)
    if not buf:
        return Response("Not found", status=404)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

# NETWORK_IMAGE_CACHE のクリーンアップ（定期的に古いキャッシュを削除）
def cleanup_image_cache() -> None:
    """イメージキャッシュをクリアする。"""
    NETWORK_IMAGE_CACHE.clear()
    logger.debug('Network image cache cleared')

# ======================================================================
# 9) Single trajectory network PNG generator (using plot_seq_matplotlib2)
# ======================================================================
def generate_pitchspeed_network_png(traj_str, pitch_seq, speed_seq, coords_override=None, xlim=None, ylim=None):
    """
    Generate a small-multiple PNG using plot_seq_matplotlib2.
    This function creates a DataFrame with a single row containing
    trajectory, coords, PitchSequence, and SpeedSequence, then calls
    plot_seq_matplotlib2 to render the pitch sequence visualization.
    
    Args:
        traj_str: trajectory string (zone sequence)
        pitch_seq: pitch category sequence (e.g., 'FFSC')
        speed_seq: speed category sequence (e.g., 'HHMM') or list of speeds
        coords_override: list of (x, z) tuples for measured plate coordinates
        xlim: tuple (xmin, xmax) for fixed x-axis range across cluster
        ylim: tuple (ymin, ymax) for fixed y-axis range across cluster
        
    Returns:
        BytesIO buffer containing PNG image, or None if generation fails
    """
    if not traj_str or len(traj_str) < 2:
        return None
    
    # If coords_override is not provided, fall back to NODE_POSITION mapping
    if coords_override is None or len(coords_override) < 2:
        # Map trajectory zones to NODE_POSITION coordinates
        coords_override = []
        for ch in traj_str:
            if ch in NODE_POSITION:
                # Use NODE_POSITION but negate x (same transform as in select_and_preprocess_data)
                x, z = NODE_POSITION[ch]
                coords_override.append((-float(x), float(z)))
            else:
                coords_override.append((0.0, 0.0))
    
    # Convert speed_seq to numeric if it's categorical (S/M/H)
    if isinstance(speed_seq, str):
        # Map S/M/H to representative speeds with larger differences for visual distinction
        # Slow: 75 mph, Medium: 88 mph, High: 98 mph (wider range for clearer size differences)
        speed_map = {'S': 75.0, 'M': 88.0, 'H': 98.0}
        speeds_numeric = [speed_map.get(ch, 88.0) for ch in speed_seq[:len(traj_str)]]
    else:
        speeds_numeric = list(speed_seq[:len(traj_str)])
    
    # Ensure all sequences have same length
    n = min(len(traj_str), len(pitch_seq), len(speeds_numeric), len(coords_override))
    if n < 2:
        return None
    
    # Create a single-row DataFrame for plot_seq_matplotlib2
    df_single = pd.DataFrame([{
        'trajectory': traj_str[:n],
        'coords': coords_override[:n],
        'PitchCategory': pitch_seq[:n],
        'PitchSpeed': speeds_numeric[:n]
    }])
    
    try:
        # Create figure for small-multiple thumbnail
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Call plot_seq_matplotlib2 with thumbnail-appropriate settings
        plot_seq_matplotlib2(
            df_vis=df_single,
            pos=0,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            legend=False,
            show_axes=False,
            square_view=True,
            square_anchor="center",
            invert_x=False,
            pad_frac=0.15
        )
        
        # Save to buffer
        buf = io.BytesIO()
        plt.tight_layout(pad=0.1)
        save_matplotlib_buffer(buf, dpi=100, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return buf
        
    except Exception as e:
        logger.exception(f'Failed to generate PNG for trajectory {traj_str}: {e}')
        plt.close('all')
        return None

# ======================================================================
# 10) 追加ユーティリティ
#     group_extraction_with_coords と plot_seq_matplotlib2
# ======================================================================

def group_extraction_with_coords(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[str]], List[List[float]]]:
    """試合日、イニング、打席番号でグループ化して連続投球シーケンスを抽出する。
    
    試合日、イニング、打席番号でグループ化して
    連続投球の zone シーケンスと plate_x plate_z の座標列を取り出す
    
    Args:
        data: zone, plate_x, plate_z, pitch_name, effective_speed 列を含むデータフレーム
        
    Returns:
        Tuple[DataFrame, pitch_types_list, pitch_speeds_list]:
            - DataFrame: trajectory と coords を持つデータフレーム
            - List[List[str]]: 各打席の球種列のリスト
            - List[List[float]]: 各打席の実測球速列のリスト
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
    df_vis: pd.DataFrame, pos: int,
    xlim: Optional[Tuple[float, float]] = None, 
    ylim: Optional[Tuple[float, float]] = None,
    strike_zone: Tuple[float, float, float, float] = (-0.83, 0.83, 1.5, 3.5),
    color_map: Optional[Dict[str, str]] = None,
    legend: bool = True,
    ax: Optional[Any] = None,
    pad_frac: float = 0.10,
    pad_x: Optional[float] = None, 
    pad_y: Optional[float] = None,
    include_strike_zone_in_bounds: bool = True,
    square_view: bool = True,
    square_anchor: str = "center",
    invert_x: bool = False,
    show_axes: bool = True,
    x_ticks: Optional[List[float]] = None, 
    y_ticks: Optional[List[float]] = None
) -> Any:
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

    # Calculate node sizes based on pitch speed (slower = smaller, faster = larger)
    spds = np.array(spds[:n], dtype=float)
    s_min, s_max = float(np.min(spds)), float(np.max(spds))
    if s_max - s_min == 0:
        sizes = np.full_like(spds, 400, dtype=float)
    else:
        # Linear mapping: 100 (slowest) to 1600 (fastest)
        sizes = 100 + 2000 * (spds - s_min) / (s_max - s_min)

    created_fig = False
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        created_fig = True

    sxmin, sxmax, symin, symax = strike_zone
    rect = plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                         fill=False, color='gray', linewidth=2, alpha=0.6)
    ax.add_patch(rect)

    # apply alpha to the scatter colors
    colors_rgba = [to_rgba(c, 0.7) for c in colors]
    ax.scatter(xs, ys, s=sizes, c=colors_rgba, edgecolors='black')

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
    #ax.set_title(f"Pitch Sequence for No.{pos}")
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
        plt.show()

    return ax

def update_elements(tapped_node: Optional[Dict[str, Any]], elements: List[Dict[str, Any]], 
                    selected_nodes: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Cytoscape内のタップされたノードに基づいて要素を更新する。
    
    Args:
        tapped_node: タップされたノード情報 ({id, ...} または None)
        elements: Cytoscape要素（ノードとエッジ）のリスト
        selected_nodes: 現在選択されたノードのリスト
        
    Returns:
        Tuple[要素のリスト, 選択ゾーンのテキスト, 更新されたノードのリスト]
    """
    if tapped_node is None:
        return elements, 'Selected Zone : ' + ', '.join(selected_nodes), selected_nodes
    
    print(selected_nodes)

    new_elements = elements.copy()
    node_id = tapped_node['id']
    print("Clicked Node:", node_id)

    # ノードが選択された順番を保持するためのリスト
    if not hasattr(update_elements, 'selected_nodes'):
       update_elements.selected_nodes = []  # 初回呼び出し時にselected_nodesリストを作成
    
    # 連続10回同じノードをタップした場合は無視（単純な誤操作対策）
    if len(selected_nodes) >= 10 and all(n == node_id for n in selected_nodes[-10:]):
        zones_text = 'Selected Zone : ' + ', '.join(selected_nodes)
        logger.debug("同一ノード連続タップを無視: %s", node_id)
        return elements, zones_text, selected_nodes
    
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

# ======================================================================
# 11) Run
# ======================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
