import pulp
import ortoolpy
import numpy as np
import pandas as pd

import time
import shutil
import pathlib
from tqdm import tqdm

from path_helper import make_log_path

def binpacking_solver(data, max_sec=60*60, options=['maxsol 1']):
    log_path = make_log_path('binpacking')
    total_weight = data.Weight.sum()
    size = len(data)
    print(f'gift num: {size}')
    print(f'total weight: {total_weight}')

    max_weight = 1000
    max_size = 100
    num_groups = max(
        int(((total_weight / max_weight)+1)*1.1),
        int(((size / max_size)+1)*1.1)
    )
    print(f'num_groups: {num_groups}')

    for i in range(1, num_groups+1):
        data[f'tourVar{str(i).zfill(2)}'] = ortoolpy.addbinvars(size)

    tourvars = [col for col in data.columns if 'tourVar' in col]
    binvars = ortoolpy.addbinvars(num_groups)
    prob = pulp.LpProblem('binpacking', pulp.LpMinimize)
    # 目的関数 ビンの数最小化
    prob += pulp.lpSum(binvars)

#     print('最大容量付与')
    for i, col in enumerate(tourvars):
        prob += pulp.lpSum(pulp.lpDot(data.Weight.values, data[col].values)) <= max_weight * binvars[i]

#     print('Giftごとに一つのビンに入れる')
    for idx in data.index:
        prob += pulp.lpSum(data.loc[idx, tourvars]) == 1

    for tvar in tourvars:
        prob += pulp.lpSum(data[tvar]) <= max_size
    
    print('最適化実行中')
    t1 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=1, keepFiles=1, maxSeconds=max_sec, threads=0, options=options))
    status = pulp.LpStatus[prob.status]
    obj_val = prob.objective.value()
    opt_time = time.time() - t1
    
    print('opt_time: ', opt_time)
    print('status: ', status)
    print('目的関数: ', obj_val)

    for col in tourvars:
        data[col] = data[col].apply(pulp.value)

    def get_group(x):
        tmp = data.loc[x][tourvars]
        return tmp[tmp==1].index.values[0]

    data['group'] = data.index.map(get_group)
    data.drop(tourvars, axis=1, inplace=True)

    gift_num_describe = data.group.value_counts().describe()
    group_weightsum_describe = data.groupby('group').Weight.sum().describe()

    log_s = pd.Series({
        'gift_num': len(data),
        'total_weight': total_weight,
        'num_groups': num_groups,
        'opt_time': opt_time,
        'status': status,
        '目的関数': obj_val
    })

    log_s.to_csv(log_path/'log_s.csv', encoding='utf-8-sig')
    data.to_csv(log_path/'data.csv', encoding='utf-8-sig')
    gift_num_describe.to_csv(log_path/'gift_num_describe.csv', encoding='utf-8-sig')
    group_weightsum_describe.to_csv(log_path/'group_weightsum_describe.csv', encoding='utf-8-sig')
    shutil.move('binpacking-pulp.mps', log_path)
    shutil.move('binpacking-pulp.sol', log_path)

    print('Gift数 要約統計量')
    print(gift_num_describe)
    print('\nグループのWeight合計 要約統計量')
    print(group_weightsum_describe)
    return data