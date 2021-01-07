import pathlib
import os
import datetime

data_path = pathlib.Path('data')

def make_log_path(foldername):
    '''
    purpose: ログフォルダ、パス作成
    args: foldername
    return: log_path
    '''
    global data_path
    assert os.path.exists(data_path), f'data_path: {data_path}がないよ'
    assert isinstance(data_path, pathlib.Path), 'data_pathはpathlib.Pathから作ってね'
    logdir = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    print(f'出力先：{foldername}/{logdir}')
    log_path = data_path / f'logs/{foldername}/{logdir}'
    os.makedirs(log_path, exist_ok=True)
    return log_path