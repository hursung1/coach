
from src.modules import Lstm, CRF, Attention
from src.slu.datareader import SLOT_PAD, y2_set, domain_set
from src.transformer import TransformerEncoder
from src.utils import load_embedding_from_npy, load_embedding_from_pkl
from preprocess.gen_embeddings_for_slu import domain2slot
import torch
from torch import nn
from torch.nn import functional as F

class BinarySLUTagger(nn.Module):
    def __init__(self, params, vocab):
        super(BinarySLUTagger, self).__init__()
        
        self.lstm = Lstm(params, vocab)
        self.num_binslot = params.num_binslot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        
    def forward(self, X, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_binslot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction, lstm_hidden
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y


class SlotNamePredictor(nn.Module):
    def __init__(self, params):
        super(SlotNamePredictor, self).__init__()
        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.enc_type = params.enc_type
        if self.enc_type == "trs":
            self.trs_enc = TransformerEncoder(input_size=self.input_dim, hidden_size=params.trs_hidden_dim, num_layers=params.trs_layers, num_heads=params.num_heads, dim_key=params.dim_key, dim_value=params.dim_value, filter_size=params.filter_size)
        elif self.enc_type == "lstm":
            self.lstm_enc = nn.LSTM(self.input_dim, params.trs_hidden_dim//2, num_layers=params.trs_layers, bidirectional=True, batch_first=True)
        
        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)
    
    def forward(self, domains, hidden_layers, binary_predictions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            hidden_states_list: list of hidden states corresponding to slots
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_predictions

        feature_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []

        bsz = domains.size()[0]
        
        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
            
            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            # get indices of B and I
            B_list = bin_label == 1
            I_list = bin_label == 2
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])

                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "trs":
                        slot_feats = self.trs_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    elif self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    # slot_feats.squeeze(0) ==> (hidden_dim)
                    feature_each_sample.append(slot_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "trs":
                        slot_feats = self.trs_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    elif self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    feature_each_sample.append(slot_feats.squeeze(0))

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[curr_index]].split("-")[1]
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                        
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            feature_list.append(feature_each_sample)
            if final_golds is not None:
                gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)

        ### predict slot names
        pred_slotname_list = []
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            
            slot_embs_based_domain = torch.FloatTensor(self.slot_embs[domain_name]).transpose(0,1).cuda()   # (emb_dim, slot_num)

            feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                pred_slotname_each_sample = torch.matmul(feature_each_sample, slot_embs_based_domain) # (num_slotname, slot_num)
            
            pred_slotname_list.append(pred_slotname_each_sample)

        if final_golds is not None:
            # only in the training mode
            return pred_slotname_list, gold_slotname_list
        else:
            return pred_slotname_list


class SentRepreGenerator(nn.Module):
    def __init__(self, params, vocab):
        super(SentRepreGenerator, self).__init__()
        self.hidden_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        
        # LSTM Encoder for template
        self.template_encoder = Lstm(params, vocab)

        # attention layers for templates and input sequences
        self.input_atten_layer = Attention(attention_size=self.hidden_size)
        self.template_attn_layer = Attention(attention_size=self.hidden_size)
        if params.tr:
            self.tr = True
            self.sr = False

        elif params.sr:
            self.sr = True
            self.tr = False 

    def forward(self, templates, tem_lengths, hidden_layers, x_lengths, y_bin=None):
        """
        Inputs:
            if template regularization:
                templates: (bsz, 3, max_template_length)
                tem_lengths: (bsz,)
            
            elif slot type regularization:
                templates == slot_type: (bsz, 3, max_slot_entities_len)
                tem_lengths == slot_type_length: (bsz, )
                y_bin: (bsz, )

            hidden_layers: (bsz, max_length, hidden_size)
            x_lengths: (bsz,)
        Outputs:
            template_sent_repre: (bsz, 3, hidden_size)
            input_sent_repre: (bsz, hidden_size)
        """
        # generate templates sentence representation
        template0 = templates[:, 0, :]
        template1 = templates[:, 1, :]
        template2 = templates[:, 2, :]

        template0_hiddens = self.template_encoder(template0)
        template1_hiddens = self.template_encoder(template1)
        template2_hiddens = self.template_encoder(template2)

        template0_repre, _ = self.template_attn_layer(template0_hiddens, tem_lengths)
        template1_repre, _ = self.template_attn_layer(template1_hiddens, tem_lengths)
        template2_repre, _ = self.template_attn_layer(template2_hiddens, tem_lengths)

        templates_repre = torch.stack((template0_repre, template1_repre, template2_repre), dim=1)  # (bsz, 3, hidden_size)

        # generate input sentence representations
        input_repre, _ = self.input_atten_layer(hidden_layers, x_lengths, y_bin)

        return templates_repre, input_repre


class SLUTagger(nn.Module):
    def __init__(self, params, vocab):
        super(SLUTagger, self).__init__()

        self.lstm = Lstm(params, vocab)
        self.num_slot = params.num_slot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_slot)
        self.crf_layer = CRF(self.num_slot)
        
    def forward(self, X, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_slot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y
        