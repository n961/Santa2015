import numpy as np
import pandas as pd
import ortoolpy
import pulp

import time
import pathlib
import shutil
import gc
import os
from tqdm import tqdm
from itertools import product

from path_helper import make_log_path


def distance(origin, destination):
    '''
    purpose: ハーバーサイン距離の計算
    args:
        origin (latitude, longitude)
        destination (latitude, longitude)
        - originとdestinationは入れ替えても同じ値を返す
    return: haversine distance
    '''
    
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371

    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) \
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return radius * c



def make_dist_matrics(data):
    '''
    purpose: 距離マトリクス作成
    args: data (pd.DataFrame, columns include [Latitude, Longitude], index=GiftId)
    return: df_dist (pd.DataFrame, columns=GiftId, index=GiftId, values=dist)
    '''
    
#     print('距離マトリクス作成')
    dist = []
    for i in data.index:
        i_dist = []
        i = data.loc[i, ['Latitude', 'Longitude']]
        for j in data.index:
            j = data.loc[j, ['Latitude', 'Longitude']]
            i_dist.append(distance(i, j))
        dist.append(i_dist)
    df_dist = pd.DataFrame(dist, index=data.index.values, columns=data.index.values)  
    return df_dist




def tsp_solver(data, log_path, max_sec=60*30, options=['maxsol 1']):
    '''
    args:
        data (GiftId, Latitude, Longitude, Weight)
        max_sec (ソルバーの最大実行時間), default 60*60
        options (ソルバー(PULP_CBC_CMD)に与えるoptions), default ['maxsol 1']
    return:
        df_val (経路の利用表)
        data (dataにver_seq(訪問順)を付与したもの)
    '''
    tid = data.TourId.values[0]
    log_path = log_path / tid
    os.makedirs(log_path, exist_ok=True)
#     initial_sol = f'data/logs/TSP/test/{tid}/TSP-pulp.sol'
#     if os.path.exists(initial_sol):
#         options.append("mips")
#         options.append(initial_sol)
    
    # df_dist の用意
    df_dist = make_dist_matrics(data)
    
    size = len(data)
    print('データ数: ', size)
    
    t1 = time.time()  

    # 距離マトリクスに対応する変数表を作成
    df_var = pd.DataFrame(index=data.index.values, columns=data.index.values)
    for col in df_var.columns:
        df_var[col] = ortoolpy.addbinvars(len(df_var))

    # 訪問順を表現する変数を付与
    data['var_seq'] = ortoolpy.addvars(size, upBound=size-1, lowBound=0, cat='Integer')
    
    prob = pulp.LpProblem('TSP', pulp.LpMinimize)

    # 目的関数 移動距離（最小化）
    prob += pulp.lpSum([pulp.lpDot(v, d) for v,d in zip(df_var.values, df_dist.values)])
    
    ## 以下制約条件付与
    # GiftId==0 の場所(北極) からスタート
    prob += data.at[0, 'var_seq'] == 0
    
    # 各地点への入次数、出次数はそれぞれ１ずつ
    for idx in df_var.index:
        prob += pulp.lpSum(df_var.loc[idx].values) == 1
        prob += pulp.lpSum(df_var[idx].values) == 1
    
    # 部分巡回路除去、ポテンシャル制約
#     print('MTZ制約付与')
    for i, j in product(df_var.index, df_var.index):
        if i == j:
            prob += df_var.at[i,j] == 0 # 同じ地点間の移動はなし
        if i==0 or j==0:
            continue
        else:
#             prob += df_var.at[i,j] + df_var.at[j,i] <= 1
            prob += (
                data.at[i, 'var_seq'] + 1
                - size * (1 - df_var.at[i,j])
                + (size - 3) * df_var.at[j,i]
                <= data.at[j, 'var_seq']
            ) # ある地点間のルートを選択したとき、逆向きルートや訪問順変数の範囲が限定される
    
    # 北極とのルートの選択により、制限できるよ
    for i in data.index:
        if i != 0:
            prob += 1 + (1 - df_var.at[0, i]) + (size - 3) * df_var.at[i, 0] <= data.at[i, 'var_seq']
            prob += data.at[i, 'var_seq'] <= (size - 1) - (1 - df_var.at[i, 0]) - (size - 3) * df_var.at[0, i]
        
    const_time = time.time()-t1
    
    
    ## 最適化実行 (中断したときもログを出力するため try節に)
    try:
        print('最適化実行中')
        t2 = time.time()
#         prob.solve(pulp.PULP_CBC_CMD(msg=1, keepFiles=1, maxSeconds=max_sec, threads=0))
        prob.solve(pulp.PULP_CBC_CMD(msg=1, keepFiles=1, maxSeconds=max_sec, options=options, fracGap=0.1, threads=0))
        flg = 'finish'
    except (KeyboardInterrupt, MemoryError, pulp.PulpSolverError) as e:
        flg = e
    finally:
        optimize_time = time.time()-t2
        gc.collect()
        print(flg)
        obj_val = prob.objective.value()
        status = pulp.LpStatus[prob.status]
        df_val = df_var.applymap(pulp.value)
        data.var_seq = data.var_seq.apply(pulp.value)
        
        ## ログ出力
        log_s = pd.Series({
            'labels': data.labels.unique().tolist(),
            'group': data.group.unique().tolist() ,
            'データ数': size,
            'max_sec': max_sec,
            'options': options,
            '制約付与時間': const_time,
            '最適化時間': optimize_time,
            'status': status,
            '目的関数': obj_val,
            'flg': flg
        })
        print(log_s)

        log_s.to_csv(log_path/'log_s.csv', encoding='utf-8-sig')
        df_dist.to_csv(log_path/'df_dist.csv', encoding='utf-8-sig')
        df_val.to_csv(log_path/'df_val.csv', encoding='utf-8-sig')
        data.to_csv(log_path/'data.csv', encoding='utf-8-sig')
        shutil.move('TSP-pulp.mps', log_path/'TSP-pulp.mps')
        try:
            shutil.move('TSP-pulp.sol', log_path/'TSP-pulp.sol')
        except:
            pass
        
    return df_val, data