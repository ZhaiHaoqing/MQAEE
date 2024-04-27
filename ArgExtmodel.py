import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, BertModel, RobertaModel
from pattern import patterns
import numpy as np

class MQAEEArgExtModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        self.generate_tagging_vocab()

        # base encoder
        if self.config.pretrained_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.base_config = BertConfig.from_pretrained(self.config.pretrained_model_path, 
                                                          cache_dir=self.config.cache_dir)
            self.base_model = BertModel.from_pretrained(self.config.pretrained_model_path, 
                                                        cache_dir=self.config.cache_dir, 
                                                        output_hidden_states=True)
        elif self.config.pretrained_model_name.startswith('roberta-'):
            self.base_config = RobertaConfig.from_pretrained(self.config.pretrained_model_path, 
                                                             cache_dir=self.config.cache_dir)
            self.base_model = RobertaModel.from_pretrained(self.config.pretrained_model_path, 
                                                           cache_dir=self.config.cache_dir, 
                                                           output_hidden_states=True)
        else:
            raise ValueError(f"pretrained_model_name is not supported.")
        
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.base_model_dim = self.base_config.hidden_size
        self.base_model_dropout = nn.Dropout(p=self.config.base_model_dropout)

        if self.config.use_history:
            self.base_model = BertWithHistoryEmbedding(self.base_model, self.base_model_dim, 2)
        self.role_label_ffn = nn.Linear(self.base_model_dim, len(self.label_stoi), bias=True)
            
    def generate_tagging_vocab(self):
        prefix = ['B', 'I']
        role_label_stoi = {'O': 0}
        for t in ["Span"]:
            for p in prefix:
                role_label_stoi['{}-{}'.format(p, t)] = len(role_label_stoi)

        self.label_stoi = role_label_stoi
        
    def get_role_seqlabels(self, roles, token_num, specify_role=None):
        labels = ['O'] * token_num
        count = 0
        for role in roles:
            start, end = role[0], role[1]
            if end > token_num:
                continue
            role_type = role[2]

            if specify_role is not None:
                if role_type != specify_role:
                    continue

            if any([labels[i] != 'O' for i in range(start, end)]):
                count += 1
                continue

            labels[start] = 'B-{}'.format("Span")
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format("Span")
                
        return labels
    
    def token_lens_to_offsets(self, token_lens):
        """Map token lengths to first word piece indices, used by the sentence
        encoder.
        :param token_lens (list): token lengths (word piece numbers)
        :return (list): first word piece indices (offsets)
        """
        max_token_num = max([len(x) for x in token_lens])
        offsets = []
        for seq_token_lens in token_lens:
            seq_offsets = [0]
            for l in seq_token_lens[:-1]:
                seq_offsets.append(seq_offsets[-1] + l)
            offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
        return offsets
    
    def token_lens_to_idxs(self, token_lens):
        """Map token lengths to a word piece index matrix (for torch.gather) and a
        mask tensor.
        For example (only show a sequence instead of a batch):
        token lengths: [1,1,1,3,1]
        =>
        indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
        masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]
        Next, we use torch.gather() to select vectors of word pieces for each token,
        and average them as follows (incomplete code):
        outputs = torch.gather(bert_outputs, 1, indices) * masks
        outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
        outputs = bert_outputs.sum(2)
        :param token_lens (list): token lengths.
        :return: a index matrix and a mask tensor.
        """
        max_token_num = max([len(x) for x in token_lens])
        max_token_len = max([max(x) for x in token_lens])
        idxs, masks = [], []
        for seq_token_lens in token_lens:
            seq_idxs, seq_masks = [], []
            offset = 0
            for token_len in seq_token_lens:
                seq_idxs.extend([i + offset for i in range(token_len)]
                                + [-1] * (max_token_len - token_len))
                seq_masks.extend([1.0 / token_len] * token_len
                                 + [0.0] * (max_token_len - token_len))
                offset += token_len
            seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
            seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
            idxs.append(seq_idxs)
            masks.append(seq_masks)
        return idxs, masks, max_token_num, max_token_len
    
    def tag_paths_to_spans(self, paths, token_nums, vocab):
        """
        Convert predicted tag paths to a list of spans (entity mentions or event
        triggers).
        :param paths: predicted tag paths.
        :return (list): a list (batch) of lists (sequence) of spans.
        """
        batch_mentions = []
        itos = {i: s for s, i in vocab.items()}
        for i, path in enumerate(paths):
            mentions = []
            cur_mention = None
            path = path.tolist()[:token_nums[i].item()]
            for j, tag in enumerate(path):
                if tag not in itos:
                    tag = 'O'
                else:
                    tag = itos[tag]
                if tag == 'O':
                    prefix = tag = 'O'
                else:
                    prefix, tag = tag.split('-', 1)
                if prefix == 'B':
                    if cur_mention:
                        mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
                elif prefix == 'I':
                    if cur_mention is None:
                        # treat it as B-*
                        cur_mention = [j, j + 1, tag]
                    elif cur_mention[-1] == tag:
                        cur_mention[1] = j + 1
                    else:
                        # treat it as B-*
                        mentions.append(cur_mention)
                        cur_mention = [j, j + 1, tag]
                else:
                    if cur_mention:
                        mentions.append(cur_mention)
                    cur_mention = None
            if cur_mention:
                mentions.append(cur_mention)
            batch_mentions.append(mentions)
        return batch_mentions
    
    def process_data(self, batch):
        enc_idxs = []
        enc_attn = []
        role_seqidxs = []
        token_lens = []
        token_nums = []
        question_lens = []
        history_idxs = []
        max_token_num = max(batch.batch_token_num)

        for tokens, pieces, trigger, arguments, question, token_len, token_num, history in zip(batch.batch_tokens, batch.batch_pieces, batch.batch_trigger, 
                                                                      batch.batch_arguments, batch.batch_question, batch.batch_token_lens, batch.batch_token_num, batch.batch_history):

            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)
            question_id = self.tokenizer.convert_tokens_to_ids(question)
            question_lens.append(len(question_id))

            enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_id + \
                        [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
            
            assert len(history) == len(piece_id)
            enc_idxs.append(enc_idx)
            enc_attn.append([1]*len(enc_idx))
            history_idxs.append([0]*(len(question_id)+2) + history + [0])
            argument_seq = self.get_role_seqlabels(arguments, len(tokens))
            token_lens.append(token_len)
            token_nums.append(token_num)
            role_seqidxs.append([self.label_stoi[s] for s in argument_seq] + [-100] * (max_token_num-len(tokens)))

        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        history_idxs = torch.LongTensor([history_idx + [0]*(max_len-len(history_idx)) for history_idx in history_idxs])
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        history_idxs = history_idxs.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, history_idxs, role_seqidxs, token_lens, torch.cuda.LongTensor(token_nums), question_lens
        
    def encode(self, piece_idxs, attention_masks, history_idxs, token_lens, question_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        if self.config.use_history:
            all_base_model_outputs = self.base_model(piece_idxs, attention_masks, history_idxs)
            # print(all_base_model_outputs)
        else:
            all_base_model_outputs = self.base_model(piece_idxs, attention_mask=attention_masks)
        base_model_outputs = all_base_model_outputs[0]

        trim_model_outputs = []
        for i, output in enumerate(base_model_outputs):
            text_token_emb = torch.cat((output[0:1, :], output[question_lens[i] + 2:, :]), 0)
            trim_model_outputs.append(text_token_emb)
        
        max_len = max([trim_model_output.size(0) for trim_model_output in trim_model_outputs])
        for i, output in enumerate(trim_model_outputs):
            padding_tensor = torch.zeros(max_len - output.size(0), output.size(1)).cuda()
            trim_model_outputs[i] = torch.cat((output, padding_tensor), dim=0)

        base_model_outputs = torch.stack(trim_model_outputs).cuda()

        if self.config.multi_piece_strategy == 'first':
            # select the first piece for multi-piece words
            offsets = self.token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            base_model_outputs = torch.gather(base_model_outputs, 1, offsets)
        elif self.config.multi_piece_strategy == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.base_model_dim) + 1
            masks = base_model_outputs.new(masks).unsqueeze(-1)
            base_model_outputs = torch.gather(base_model_outputs, 1, idxs) * masks
            base_model_outputs = base_model_outputs.view(batch_size, token_num, token_len, self.base_model_dim)
            base_model_outputs = base_model_outputs.sum(2)
        else:
            raise ValueError(f'Unknown multi-piece token handling strategy: {self.config.multi_piece_strategy}')
        base_model_outputs = self.base_model_dropout(base_model_outputs)
        return base_model_outputs

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, history_idxs, role_seqidxs, token_lens, token_nums, question_lens = self.process_data(batch)
        
        # encoding
        base_model_outputs = self.encode(enc_idxs, enc_attn, history_idxs, token_lens, question_lens)
        logits = self.role_label_ffn(base_model_outputs)
        preds = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), role_seqidxs.view(-1))
    
        return loss
    
    def predict(self, batch, batch_pred_triggers=None):
        assert len(batch.batch_tokens) == 1
        
        self.eval()
        with torch.no_grad():
            enc_idxs = []
            enc_attn = []
            token_lens = []
            token_nums = []
            question_lens = []

            question = "[trigger] <pos>[trigger_position]</pos>, [event_type], [argument_role]?"

            token_start = np.cumsum(batch.batch_token_lens[0])
            token_start = np.insert(token_start, 0, 0)
            history_idx = np.zeros((len(batch.batch_pieces[0]), ), dtype=int)
            
            tokens, pieces, triggers, token_len, token_num  = batch.batch_tokens[0], batch.batch_pieces[0], batch_pred_triggers[0], batch.batch_token_lens[0], batch.batch_token_num[0]
                
            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)

            for trigger in triggers:
                start, end, event_type = trigger[0], trigger[1], trigger[2]
                text_span = ' '.join(tokens[start: end])
                new_question = question.replace("[trigger]", text_span).replace("[trigger_position]", str(start)).replace("[event_type]", event_type)

                arg_roles = patterns[self.config.dataset][event_type]

                for arg_role in arg_roles:
                    new_question_arg = new_question.replace("[argument_role]", arg_role)
                    question_tokens = self.tokenizer.tokenize(new_question_arg)
                    question_id = self.tokenizer.convert_tokens_to_ids(question_tokens)
                    question_lens.append(len(question_id))

                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]

                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    token_lens.append(token_len)
                    token_nums.append(token_num)
            
            if len(enc_idxs) == 0:
                self.train() 
                return [[] for i in range(len(batch_pred_triggers))]
            
            batch_pred_arguments = []
            if self.config.use_history:
                preds = []
                for idx, attn, token_len, question_len, token_num in zip(enc_idxs, enc_attn, token_lens, question_lens, token_nums):
                    history = [0] * (question_len + 2) + history_idx.tolist() + [0]
                    history = torch.LongTensor([history]).cuda()
                    
                    idx = torch.LongTensor([idx]).cuda()
                    attn = torch.LongTensor([attn]).cuda()
                    
                    token_len = [token_len]
                    question_len = [question_len]
                    token_num = torch.LongTensor([token_num]).cuda()
                    
                    base_model_output = self.encode(idx, attn, history, token_len, question_len)
                    logits = self.role_label_ffn(base_model_output)
                    pred = logits.argmax(dim=-1)
                    pred = self.tag_paths_to_spans(pred, token_num, self.label_stoi)
                    
                    preds.append(pred[0])
                    for pred_arg in pred[0]:
                        pred_arg_start, pred_arg_end = pred_arg[0], pred_arg[1]
                        history_idx[token_start[pred_arg_start]: token_start[pred_arg_end]] = 1
                    
            else:
                max_len = max([len(enc_idx) for enc_idx in enc_idxs])
                enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
                enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
                enc_idxs = enc_idxs.cuda()
                enc_attn = enc_attn.cuda()
                token_nums = torch.cuda.LongTensor(token_nums)
                base_model_outputs = self.encode(enc_idxs, enc_attn, None, token_lens, question_lens)
                logits = self.role_label_ffn(base_model_outputs)
                preds = logits.argmax(dim=-1)
                preds = self.tag_paths_to_spans(preds, token_nums, self.label_stoi)

            i = 0
            for triggers in batch_pred_triggers:
                for trigger in triggers:
                    pred_arguments = []
                    arg_roles = patterns[self.config.dataset][trigger[2]]
                    for arg_role in arg_roles:
                        for j in range(len(preds[i])):
                            start, end = preds[i][j][0], preds[i][j][1]
                            pred_arguments.append((start, end, arg_role))
                        i += 1
                    batch_pred_arguments.append(pred_arguments)

        self.train()
        return batch_pred_arguments
    

class BertWithHistoryEmbedding(nn.Module):
    def __init__(self, base_model, history_embedding_dim, history_embedding_vocab_size=2):
        super(BertWithHistoryEmbedding, self).__init__()
        
        self.bert = base_model
        self.token_embedding = self.bert.embeddings.word_embeddings
        self.history_embedding = nn.Embedding(history_embedding_vocab_size, history_embedding_dim)
        self.history_embedding.weight.data.uniform_(-0.02, 0.02)

    def forward(self, input_ids, attention_mask, history_ids):

        token_embed = self.token_embedding(input_ids)
        # print(token_embed)
        history_embed = self.history_embedding(history_ids)
        # print(history_embed)
        
        final_embed = token_embed + history_embed
        
        outputs = self.bert(inputs_embeds=final_embed, attention_mask=attention_mask)
        
        return outputs