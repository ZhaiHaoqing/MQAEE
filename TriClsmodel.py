import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, BertModel, RobertaModel
from pattern import patterns
import numpy as np

class MQAEETriClfModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        self.label_list = [x for x in sorted(self.type_set["trigger"])]

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
        
        self.label_ffn = nn.Linear(self.base_model_dim, 2, bias=False)
    
    def process_data(self, batch):
        sent_res = self.tokenizer(batch.batch_questions, padding=True, return_tensors="pt")
        sent_input_ids = sent_res["input_ids"].cuda()
        sent_attention_mask = sent_res["attention_mask"].cuda()
        
        labels = torch.cuda.LongTensor(batch.batch_labels)
                
        return sent_input_ids, sent_attention_mask, labels
    
    def embed(self, input_ids, attention_mask):        
        outputs = self.base_model(input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = outputs["pooler_output"]
        return embeddings
    
    def forward(self, batch):
        question_input_ids, question_attention_mask, labels = self.process_data(batch)
        
        question_embeddings = self.embed(question_input_ids, question_attention_mask)
        logits = self.label_ffn(question_embeddings)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss
    
    def predict(self, batch, batch_pred_spans):
        self.eval()
        with torch.no_grad():
            enc_idxs = []
            enc_attn = []
            question = "The trigger word is [trigger] <pos>[trigger_position]</pos>, [event_type], [argument_roles]?"

            for pred_spans, pieces, tokens in zip(batch_pred_spans, batch.batch_pieces, batch.batch_tokens):
                piece_id = self.tokenizer.convert_tokens_to_ids(pieces)

                for span in pred_spans:

                    start, end = span[0], span[1]
                    tri_pos = str(start)
                    text_span = ' '.join(tokens[start:end])
                    new_question = question.replace("[trigger]", text_span).replace("[trigger_position]", tri_pos)
                    
                    for event_type in self.label_list:
                        arg_roles = ', '.join(sorted(patterns[self.config.dataset][event_type].keys()))
                        _new_question = new_question.replace("[event_type]", event_type).replace("[argument_roles]", arg_roles)
                        
                        question_pieces = self.tokenizer.tokenize(_new_question)
                        question_idx = self.tokenizer.convert_tokens_to_ids(question_pieces)
                        question_idx = question_idx[:self.config.max_question_length]

                        enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_idx + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                        
                        enc_idxs.append(enc_idx)
                        enc_attn.append([1]*len(enc_idx))

            if len(enc_idxs) == 0:
                return [[] for _ in range(len(batch_pred_spans))]
            
            max_len = max([len(enc_idx) for enc_idx in enc_idxs])
            enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
            enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
            enc_idxs = enc_idxs.cuda()
            enc_attn = enc_attn.cuda()

            question_embeddings = self.embed(enc_idxs, enc_attn)
            logits = self.label_ffn(question_embeddings)
            preds = logits.argmax(dim=-1).view(-1, len(self.label_list))

            batch_pred_triggers = []
            idx = 0
            for pred_spans in batch_pred_spans:
                pred_triggers = []
                for pred_span in pred_spans:
                    for i, pred in enumerate(preds[idx]):
                        if pred == 1:
                            pred_triggers.append((pred_span[0], pred_span[1], self.label_list[i]))
                    idx += 1
                batch_pred_triggers.append(pred_triggers)

        self.train()

        return batch_pred_triggers

