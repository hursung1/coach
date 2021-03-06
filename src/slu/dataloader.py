
from src.slu.datareader import datareader, PAD_INDEX
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger()


class Dataset(data.Dataset):
    def __init__(self, X, y1, y2, domains, list=None, learn_mode=0):
        """
        For no regularization (learn_mode=0)
        - list = None
        For template regularization (learn_mode=1)
        - list: template list
        For slot regularization (learn_mode=2)
        - list: slot type list
        """
        self.learn_mode = learn_mode
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.domains = domains
        if self.learn_mode == 1: # template regularization
            self.template_list = list
        elif self.learn_mode == 2: # slot regularization
            self.slot_type_list = list
        elif self.learn_mode == 0:
            pass
        else:
            print("Wrong learn_mode input")
            exit()

    def __getitem__(self, index):
        if self.learn_mode == 1:
            return self.X[index], self.y1[index], self.y2[index], self.domains[index], self.template_list[index]
        elif self.learn_mode == 2:
            return self.X[index], self.y1[index], self.y2[index], self.domains[index], self.slot_type_list[index]
        else:
            return self.X[index], self.y1[index], self.y2[index], self.domains[index]
    
    def __len__(self):
        return len(self.X)


def collate_fn_for_label_encoder(data):
    X, y1, y2, domains, templates = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    tem_lengths = [len(sample_tem[0]) for sample_tem in templates]
    max_tem_len = max(tem_lengths)
    padded_templates = torch.LongTensor(len(templates), 3, max_tem_len).fill_(PAD_INDEX)
    for j, sample_tem in enumerate(templates):
        length = tem_lengths[j]
        padded_templates[j, 0, :length] = torch.LongTensor(sample_tem[0])
        padded_templates[j, 1, :length] = torch.LongTensor(sample_tem[1])
        padded_templates[j, 2, :length] = torch.LongTensor(sample_tem[2])
    tem_lengths = torch.LongTensor(tem_lengths)
    
    return padded_seqs, lengths, y1, y2, domains, padded_templates, tem_lengths
    
def collate_fn_for_se_reg(data):
    X, y1, y2, domains, slot_type = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    st_lengths = [len(st[0]) for st in slot_type] # the number of 'slot types' for each speech    
    max_st_len = max(st_lengths)
    padded_slot_type = torch.LongTensor(len(slot_type), 3, max_st_len).fill_(PAD_INDEX)

    for j, sample_st in enumerate(slot_type):
        length = st_lengths[j]
        padded_slot_type[j, 0, :length] = torch.LongTensor(sample_st[0])
        padded_slot_type[j, 1, :length] = torch.LongTensor(sample_st[1])
        padded_slot_type[j, 2, :length] = torch.LongTensor(sample_st[2])
    st_lengths = torch.LongTensor(st_lengths)
    
    return padded_seqs, lengths, y1, y2, domains, padded_slot_type, st_lengths
    

def collate_fn(data):
    X, y1, y2, domains = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    return padded_seqs, lengths, y1, y2, domains


def get_dataloader(tgt_domain, batch_size, use_label_encoder, enable_sr, n_samples):
    all_data, vocab = datareader(use_label_encoder, enable_sr)
    if use_label_encoder:
        train_data = {"utter": [], "y1": [], "y2": [], "domains": [], "template_list": []}
    elif enable_sr:
        train_data = {"utter": [], "y1": [], "y2": [], "domains": [], "slot_type_list": []}
    else:
        train_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    for dm_name, dm_data in all_data.items():
        if dm_name != tgt_domain:
            train_data["utter"].extend(dm_data["utter"])
            train_data["y1"].extend(dm_data["y1"])
            train_data["y2"].extend(dm_data["y2"])
            train_data["domains"].extend(dm_data["domains"])

            if use_label_encoder:
                train_data["template_list"].extend(dm_data["template_list"])
            elif enable_sr:
                train_data["slot_type_list"].extend(dm_data["slot_type_list"])

    val_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    test_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    if n_samples == 0:
        # first 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][:500]

        # the rest as test set
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]    
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]      # rest as test set
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]      # rest as test set
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]    # rest as test set

    else:
        # first n samples as train set
        train_data["utter"].extend(all_data[tgt_domain]["utter"][:n_samples])
        train_data["y1"].extend(all_data[tgt_domain]["y1"][:n_samples])
        train_data["y2"].extend(all_data[tgt_domain]["y2"][:n_samples])
        train_data["domains"].extend(all_data[tgt_domain]["domains"][:n_samples])

        if use_label_encoder:
            train_data["template_list"].extend(all_data[tgt_domain]["template_list"][:n_samples])
        elif enable_sr:
            train_data["slot_type_list"].extend(all_data[tgt_domain]["slot_type_list"][:n_samples])

        # from n to 500 samples as validation set
        val_data["utter"] = all_data[tgt_domain]["utter"][n_samples:500]  
        val_data["y1"] = all_data[tgt_domain]["y1"][n_samples:500]
        val_data["y2"] = all_data[tgt_domain]["y2"][n_samples:500]
        val_data["domains"] = all_data[tgt_domain]["domains"][n_samples:500]

        # the rest as test set (same as zero-shot)
        test_data["utter"] = all_data[tgt_domain]["utter"][500:]
        test_data["y1"] = all_data[tgt_domain]["y1"][500:]
        test_data["y2"] = all_data[tgt_domain]["y2"][500:]
        test_data["domains"] = all_data[tgt_domain]["domains"][500:]

    if use_label_encoder:
        dataset_tr = Dataset(train_data["utter"], train_data["y1"], train_data["y2"], train_data["domains"], train_data["template_list"], learn_mode=1)
    elif enable_sr:
        dataset_tr = Dataset(train_data["utter"], train_data["y1"], train_data["y2"], train_data["domains"], train_data["slot_type_list"], learn_mode=2)
    else:
        dataset_tr = Dataset(train_data["utter"], train_data["y1"], train_data["y2"], train_data["domains"])
    dataset_val = Dataset(val_data["utter"], val_data["y1"], val_data["y2"], val_data["domains"])
    dataset_test = Dataset(test_data["utter"], test_data["y1"], test_data["y2"], test_data["domains"])

    if use_label_encoder:
        dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_label_encoder)
    elif enable_sr:
        dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_se_reg)
    else:
        dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, vocab