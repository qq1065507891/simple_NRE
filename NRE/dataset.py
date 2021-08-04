import torch

from torch.utils.data import Dataset

from NRE.utils import load_pkl


class CustomDataset(Dataset):
    def __init__(self, file_path, name):
        self.data = load_pkl(file_path, name)

    def __getitem__(self, item):
        sample = self.data[item]
        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lens = [len(data[0]) for data in batch]
    max_len = max(lens)

    sent_list = []
    head_pos_list = []
    tail_pos_list = []
    mask_pos_list = []
    relation_list = []

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    for data in batch:
        sent, head_pos, tail_pos, mask_pos, relation = data
        sent_list.append(_padding(sent, max_len))
        head_pos_list.append(_padding(head_pos, max_len))
        tail_pos_list.append(_padding(tail_pos, max_len))
        mask_pos_list.append(_padding(mask_pos, max_len))
        relation_list.append(relation)

    return torch.tensor(sent_list), torch.tensor(head_pos_list), torch.tensor(tail_pos_list),\
           torch.tensor(mask_pos_list), torch.tensor(relation_list)
