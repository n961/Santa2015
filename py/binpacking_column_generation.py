import time
import gc
import pulp
import ortoolpy
import shutil

from path_helper import make_log_path

# 列生成法
def solve_binpacking(max_weight, data):
    log_path = make_log_path('binpacking_test')
    
    comb_pattern = [[idx] for idx in data.index]
    cnt = 1
    while True:
        print(f'\n-------{cnt}週目-------')
        gc.collect()
        zcnt = str(cnt).zfill(3)
        name_dual = f'{zcnt}_dual'
        name_kp = f'{zcnt}_KP'
        
        data = solve_bp_dual(max_weight, data, comb_pattern, name_dual)
        kp_solved, kp_val, a = solve_KP(max_weight, data, name_kp)
        print(f'\t kp_val: {kp_val}')
        shutil.move(f'{name_dual}-pulp.mps', log_path)
        shutil.move(f'{name_dual}-pulp.sol', log_path)
        shutil.move(f'{name_kp}-pulp.mps', log_path)
        shutil.move(f'{name_kp}-pulp.sol', log_path)   
        
        if (not kp_solved) or (kp_val <= 15):
            print('break')
            break
        comb_pattern.append(data.query('kp_var>0').index.tolist())
        cnt += 1

    primal_var = [pulp.LpVariable(f'x{i+1}', lowBound=0) for i in range(len(comb_pattern))]
    prob = pulp.LpProblem('primal', pulp.LpMinimize)
    prob += pulp.lpSum(primal_var)
    for i, comb in enumerate(comb_pattern):
        if data.loc[comb, 'dual_var'].sum() != 1:
            prob += primal_var[i] == 0
    for idx in data.index:
        prob += pulp.lpSum([primal_var[i] for i, comb in enumerate(comb_pattern) if idx in comb]) == 1
        
    print('start solving primal problem!!!!!!')
    t1 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=1,keepFiles=1, threads=0))
    print(f'took {round((time.time()-t1), 3)} sec to solve primal')
    shutil.move('primal-pulp.mps', log_path)
    shutil.move('primal-pulp.sol', log_path)
    data.to_csv(log_path/'data.csv', encoding='utf-8-sig')
    primal_val = list(map(pulp.value, primal_var))
    ret = pd.DataFrame(columns=['comb_pattern', 'primal_val'], index=range(len(comb_pattern)))
    ret['comb_pattern'] = comb_pattern
    ret['primal_val'] = primal_val
    
    return ret

# 双対問題        
def solve_bp_dual(max_weight, data, comb_pattern, name):
    data['dual_var'] = ortoolpy.addvars(len(data), lowBound=0)
    prob = pulp.LpProblem(name, pulp.LpMaximize)
    prob += pulp.lpSum(data.dual_var)
    for comb in comb_pattern:
        prob += pulp.lpSum(data.loc[comb, 'dual_var']) <= 1
#     print('\t start solving dual')
    t1 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=1,keepFiles=1, threads=0))
    print(f'\t dual: {round((time.time()-t1), 3)} sec')
    prob.roundSolution()
    data.dual_var = data.dual_var.apply(pulp.value)
    obj_val = prob.objective.value()
    return data

# ナップサック問題
def solve_KP(max_weight, data, name):
    data['kp_var'] = ortoolpy.addbinvars(len(data))
    prob = pulp.LpProblem(name, pulp.LpMaximize)
    prob += pulp.lpSum(pulp.lpDot(data.dual_var, data.kp_var))
    prob += pulp.lpSum(pulp.lpDot(data.Weight, data.kp_var)) <= max_weight
#     print('\t start solving knapsack')
    t1 = time.time()
    # fracGap指定なしの場合、87週目で進まなくなる
    prob.solve(pulp.PULP_CBC_CMD(msg=1, keepFiles=1, options = ['maxsol 5'], fracGap=0.1, threads=0))
    print(f'\t KP: {round((time.time()-t1), 3)} sec')
    obj_val = prob.objective.value()
    data.kp_var = data.kp_var.apply(pulp.value)
    solved = prob.status == 1
    return solved, obj_val, data