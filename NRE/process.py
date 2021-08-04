import csv
import os
import codecs

import jieba

from NRE.utils import load_data, ensure_dir, save_pkl
from NRE.config import config
from NRE.vocab import Vocab


def relation_tokenize(relations, file):
    """
    获取关系编码
    :param relations:
    :param file:
    :return:
    """
    relations_list = []
    relations_dict = {}
    out = []

    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            relations_list.append(line.strip())

    for i, rel in enumerate(relations_list):
        relations_dict[rel] = i

    for rel in relations:
        out.append(relations_dict[rel])

    return out


def get_mask_feature(sent_len, entities_pos):
    """
    获取mask编码
    :param entities_pos:
    :param sen_len:
    :return:
    """
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sent_len - entities_pos[1])
    return left + middle + right


def get_pos_feature(sent_len, entities_pos, entity_len, pos_limit):
    """
    获取位置编码
    :param sent_len:
    :param entities_pos:
    :param entity_len:
    :param pos_limit:
    :return:
    """
    left = list(range(-entities_pos, 0))
    middle = [0] * entity_len
    right = list(range(1, sent_len - entities_pos - entity_len +1))
    pos = left + middle + right

    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i] = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]

    return pos


def build_data(raw_data, vocab):
    sents = []
    head_pos = []
    tail_pos = []
    mask_pos = []

    if vocab.name == 'word':
        for data in raw_data:
            sent = [vocab.word2id.get(w, 1) for w in data[-2]]
            pos = list(range(len(sent)))
            head, tail = int(data[-1][0]), int(data[-1][-1])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = get_pos_feature(len(sent), head, 1, config.pos_limit)
            tail_p = get_pos_feature(len(sent), tail, 1, config.pos_limit)
            mask_p = get_mask_feature(len(sent), entities_pos)
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    else:
        for data in raw_data:
            sent = [vocab.word2id.get(w, 1) for w in data[0]]
            head, tail = int(data[3]), int(data[6])
            head_len, tail_len = len(data[1]), len(data[4])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = get_pos_feature(len(sent), head, head_len, config.pos_limit)
            tail_p = get_pos_feature(len(sent), tail, tail_len, config.pos_limit)
            mask_p = get_mask_feature(len(sent), entities_pos)
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    return sents, head_pos, tail_pos, mask_pos


def build_vocab(raw_data, out_path):
    if config.word_segment:
        vocab = Vocab('word')
        for data in raw_data:
            vocab.add_sentences(data[-2])
    else:
        vocab = Vocab('char')
        for data in raw_data:
            vocab.add_sentences(data[0])
    vocab.trim(config.min_freq)

    ensure_dir(out_path)

    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    save_pkl(vocab_path, vocab, 'vocab')

    with codecs.open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2id.keys()]))

    return vocab, vocab_path


def split_sentences(datas):
    new_data_sentence = []
    jieba.add_word('HEAD')
    jieba.add_word('TAIL')

    for sentence in datas.values:
        head, tail = sentence[3], sentence[6]
        new_sent = sentence[0].replace(sentence[2], 'HEAD', 1)
        new_sent = new_sent.replace(sentence[5], 'TAIL', 1)
        new_sent = jieba.lcut(new_sent)
        head_pos, tail_pos = new_sent.index('HEAD'), new_sent.index('TAIL')
        new_sent[head_pos] = head
        new_sent[tail_pos] = tail
        sentence = sentence.tolist()
        sentence.append(new_sent)
        sentence.append([head_pos, tail_pos])
        new_data_sentence.append(sentence)
    return new_data_sentence


def exist_relation(file, file_type):
    with codecs.open(file, encoding='utf-8') as f:
        if file_type == 'csv':
            f = csv.DictReader(f)
        for line in f:
            keys = list(line.keys())
            try:
                num = keys.index('relation')
            except:
                num = -1
            return num


def process(input_path, out_path, file_type):
    print('数据预处理开始')
    file_type = file_type.lower()
    # assert file_type in ['csv', 'json']

    print('加载原始数据')

    train_fp = os.path.join(input_path, 'train.' + file_type)
    test_fp = os.path.join(input_path, 'test.' + file_type)
    relation_fp = os.path.join(input_path, 'relation.txt')

    # relation_place = exist_relation(train_fp, file_type)

    train_data = load_data(train_fp)
    test_data = load_data(test_fp)

    train_relations = train_data['relation']
    test_relations = test_data['relation']

    if config.is_chinese and config.word_segment:
        train_data = split_sentences(train_data)
        test_data = split_sentences(test_data)

    print('构建词典')
    vocab, vocab_path = build_vocab(train_data, out_path)

    print('构建模型数据')
    train_sents, train_head_pos, train_tail_pos, train_mask = build_data(train_data, vocab)
    test_sents, test_head_pos, test_tail_pos, test_mask = build_data(test_data, vocab)

    print('构建关系型数据')
    train_relations_token = relation_tokenize(train_relations, relation_fp)
    test_relations_token = relation_tokenize(test_relations, relation_fp)

    ensure_dir(out_path)
    train_data = list(
        zip(train_sents, train_head_pos, train_tail_pos, train_mask, train_relations_token)
    )

    test_data = list(
        zip(test_sents, test_head_pos, test_tail_pos, test_mask, test_relations_token)
    )

    train_data_path = os.path.join(out_path, 'train.pkl')
    test_data_path = os.path.join(out_path, 'test.pkl')

    save_pkl(train_data_path, train_data, 'train_data')
    save_pkl(test_data_path, test_data, 'test_data')

    print('数据预处理结束')
