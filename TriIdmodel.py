import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, BertModel, RobertaModel

class MQAEETriIdModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
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
        self.base_model_dropout = nn.Dropout(p=self.config.span_base_model_dropout)
        
        self.trigger_label_ffn = nn.Linear(self.base_model_dim, len(self.label_stoi), bias=True)
            
    def generate_tagging_vocab(self):
        prefix = ['B', 'I']
        trigger_label_stoi = {'O': 0}
        for t in ["Span"]:
            for p in prefix:
                trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

        self.label_stoi = trigger_label_stoi
    
    def get_span_seqlabels(self, spans, token_num, specify_span=None):
        labels = ['O'] * token_num
        count = 0
        for span in spans:
            start, end = span[0], span[1]
            if end > token_num:
                continue
            span_type = span[2]

            if specify_span is not None:
                if span_type != specify_span:
                    continue

            if any([labels[i] != 'O' for i in range(start, end)]):
                count += 1
                continue

            labels[start] = 'B-{}'.format(span_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(span_type)
                
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
        trigger_seqidxs = []
        token_lens = []
        token_nums = []
        max_token_num = max(batch.batch_token_num)

        question = "Which word is the trigger word?"

        question_tokens = self.tokenizer.tokenize(question)
        question_id = self.tokenizer.convert_tokens_to_ids(question_tokens)
        
        for tokens, pieces, spans, token_len, token_num in zip(batch.batch_tokens, batch.batch_pieces, batch.batch_spans, 
                                                                      batch.batch_token_lens, batch.batch_token_num):
            
            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)
            enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_id + \
                        [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
            
            enc_idxs.append(enc_idx)
            enc_attn.append([1]*len(enc_idx))  
            
            trigger_seq = self.get_span_seqlabels(spans, len(tokens))
            token_lens.append(token_len)
            token_nums.append(token_num)
            trigger_seqidxs.append([self.label_stoi[s] for s in trigger_seq] + [-100] * (max_token_num-len(tokens)))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        trigger_seqidxs = torch.cuda.LongTensor(trigger_seqidxs)
        return enc_idxs, enc_attn, trigger_seqidxs, token_lens, torch.cuda.LongTensor(token_nums), len(question_tokens)
    
    def encode(self, piece_idxs, attention_masks, token_lens, question_token_len):
        batch_size, _ = piece_idxs.size()
        all_base_model_outputs = self.base_model(piece_idxs, attention_mask=attention_masks)
        base_model_outputs = all_base_model_outputs[0]

        text_token_embs = torch.cat((base_model_outputs[:, 0:1, :], base_model_outputs[:, question_token_len + 2:, :]), 1)
        
        if self.config.multi_piece_strategy == 'first':
            # select the first piece for multi-piece words
            offsets = self.token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            text_token_embs = torch.gather(text_token_embs, 1, offsets)
        elif self.config.multi_piece_strategy == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.base_model_dim) + 1 # [batch_size, y, hidden_size]
            masks = text_token_embs.new(masks).unsqueeze(-1)
            text_token_embs = torch.gather(text_token_embs, 1, idxs) * masks # [batch_size, y, hidden_size]
            text_token_embs = text_token_embs.view(batch_size, token_num, token_len, self.base_model_dim)
            text_token_embs = text_token_embs.sum(2)
        else:
            raise ValueError(f'Unknown multi-piece token handling strategy: {self.config.multi_piece_strategy}')
        text_token_embs = self.base_model_dropout(text_token_embs)
        return text_token_embs
    
    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, trigger_seqidxs, token_lens, token_nums, question_token_len = self.process_data(batch)

        # encoding
        base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens, question_token_len)
        logits = self.trigger_label_ffn(base_model_outputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), trigger_seqidxs.view(-1))
        
        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, token_lens, token_nums, question_token_len = self.process_data(batch)
            
            # encoding
            base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens, question_token_len)
            logits = self.trigger_label_ffn(base_model_outputs)
            preds = logits.argmax(dim=-1)
            pred_triggers = self.tag_paths_to_spans(preds, token_nums, self.label_stoi)
        
        self.train()
        return pred_triggers