import os, sys, logging, tqdm, pprint
import torch
import numpy as np
from collections import namedtuple
from transformers import RobertaTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import Adam
from TriIdmodel import MQAEETriIdModel
from scorer import compute_ED_scores, print_scores

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_text', 'batch_triggers', 'batch_spans']
EDBatch = namedtuple('EDBatch', field_names=EDBatch_fields, defaults=[None] * len(EDBatch_fields))

def ED_collate_fn(batch):
    return EDBatch(
        batch_doc_id=[instance["doc_id"] for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance["tokens"] for instance in batch], 
        batch_pieces=[instance["pieces"] for instance in batch], 
        batch_token_lens=[instance["token_lens"] for instance in batch], 
        batch_token_num=[instance["token_num"] for instance in batch], 
        batch_text=[instance["text"] for instance in batch], 
        batch_triggers=[instance["triggers"] for instance in batch],
        batch_spans=[instance["spans"] for instance in batch],
    )

class MQAEETriIdTrainer(object):
    def __init__(self, config, type_set=None):
        self.tokenizer = None
        self.config = config
        self.type_set = type_set
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            span_state = torch.load(os.path.join(checkpoint, "best_span_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = span_state["tokenizer"]
            self.type_set = span_state["type_set"]
            self.span_model = MQAEETriIdModel(self.config, self.tokenizer)
            self.span_model.load_state_dict(span_state['span_model'])
            self.span_model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_path, cache_dir=self.config.cache_dir, do_lower_case=False)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_path, cache_dir=self.config.cache_dir, do_lower_case=False)
            
            self.span_model = MQAEETriIdModel(self.config, self.tokenizer)
            self.span_model.cuda(device=self.config.gpu_device)
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Removing overlapping triggers and over-length examples")
        
        # greedily remove overlapping triggers
        n_total = 0
        new_data = []
        for dt in data:
            
            n_total += 1
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
            no_overlap_flag = np.ones((len(dt["tokens"]), ), dtype=bool)
            spans = []
            for trigger in dt["triggers"]:
                start, end = trigger[0], trigger[1]
                # TODO: whether to filter 
                if end - start > 1:
                    continue
                # if np.all(no_overlap_flag[start:end]):
                #     spans.append((trigger[0], trigger[1], "Span"))
                #     no_overlap_flag[start:end] = False
                spans.append((trigger[0], trigger[1], "Span"))
            
            pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            
            new_dt = {"doc_id": dt["doc_id"], 
                      "wnd_id": dt["wnd_id"], 
                      "tokens": dt["tokens"], 
                      "pieces": [p for w in pieces for p in w], 
                      "token_lens": token_lens, 
                      "token_num": len(dt["tokens"]), 
                      "text": dt["text"], 
                      "triggers": dt["triggers"],
                      "spans": spans,
                     }
            
            new_data.append(new_dt)
                
        logger.info(f"There are {len(new_data)}/{n_total} ED instances after removing overlapping triggers and over-length examples")

        return new_data

    def train(self, train_data, dev_data):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_dev_data = self.process_data(dev_data)

        # train span identification
        train_batch_num = len(internal_train_data) // self.config.span_train_batch_size + (len(internal_train_data) % self.config.span_train_batch_size != 0)
        span_param_groups = [
            {
                'params': [p for n, p in self.span_model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.span_base_model_learning_rate, 'weight_decay': self.config.span_base_model_weight_decay
            },
            {
                'params': [p for n, p in self.span_model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.span_learning_rate, 'weight_decay': self.config.span_weight_decay
            },
        ]
        span_optimizer = Adam(params=span_param_groups)
        span_scheduler = get_linear_schedule_with_warmup(span_optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.span_warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.span_max_epoch)
        
        span_best_scores = {"trigger_id": {"f1": 0.0}}
        span_best_epoch = -1

        for epoch in range(1, self.config.span_max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train Span {}'.format(epoch))
            
            self.span_model.train()
            span_optimizer.zero_grad()
            cummulate_span_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.span_train_batch_size // self.config.span_accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                
                span_loss = self.span_model(batch)
                span_loss = span_loss * (1 / self.config.span_accumulate_step)
                cummulate_span_loss.append(span_loss.item())
                span_loss.backward()

                if (batch_idx + 1) % self.config.span_accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.span_model.parameters(), self.config.span_grad_clipping)
                    span_optimizer.step()
                    span_scheduler.step()
                    span_optimizer.zero_grad()
                    
            progress.close()
            logger.info(f"Average training span loss: {np.mean(cummulate_span_loss)}")
            
            span_predictions = self.internal_span_predict(internal_dev_data, split="Dev Span")
            dev_scores = compute_ED_scores(span_predictions, internal_dev_data, metrics={"trigger_id", "trigger_cls"})

            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["trigger_id"]["f1"] >= span_best_scores["trigger_id"]["f1"]:
                logger.info("Saving best span model")
                span_state = dict(span_model=self.span_model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(span_state, os.path.join(self.config.output_dir, "best_span_model.state"))
                span_best_scores = dev_scores
                span_best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"span_best_epoch": span_best_epoch, "span_best_scores": span_best_scores}))
        
    def internal_span_predict(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.span_eval_batch_size + (len(eval_data) % self.config.span_eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.span_eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_pred_spans = self.span_model.predict(batch)
            for doc_id, wnd_id, tokens, text, pred_spans in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_pred_spans):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "triggers": pred_spans
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions
    
    def predict(self, data, **kwargs):
        assert self.tokenizer and self.span_model
        internal_data = self.process_data(data)
        predictions = self.internal_span_predict(internal_data, split="Test")
        return predictions