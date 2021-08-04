import codecs
import os
import torch
import random
import pickle
import numpy as np
import pandas as pd


def ensure_dir(path):
    """
    确保文件夹存在
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def load_data(path):
    data = pd.read_csv(path)
    return data


def save_pkl(path, obj, obj_name):
    print(f'save {obj_name} in {path}')
    with codecs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path, name):
    print(f'load {name} in {path}')
    with codecs.open(path , 'rb') as f:
        data = pickle.load(f)
    return data


def get_result(entity1, entity2, key):
    """
    国籍
    祖籍
    导演
    出生地
    主持人
    所在城市
    所属专辑
    连载网站
    出品公司
    毕业院校
    :param key:
    :return:
    """
    relations = {
        "0": "国籍",
        "1": "祖籍",
        "2": "导演",
        "3": "出生地",
        "4":" 主持人",
        "5":"所在城市",
        "6": "所属专辑",
        "7": "连载网站",
        "8": "出品公司",
        "9": "毕业院校"
    }

    result = {
        'entity1': entity1,
        'entity2': entity2,
        'relation': relations.get(str(key))
    }

    return result
