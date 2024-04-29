import os, sys, logging, tqdm, pprint, json
import torch
import numpy as np
from collections import namedtuple
from transformers import RobertaTokenizer, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import Adam
from TriIdmodel import MQAEETriIdModel
from TriClsmodel import MQAEETriClfModel
from scorer import compute_ED_scores, print_scores
from pattern import patterns
from argparse import Namespace

logger = logging.getLogger(__name__)

EDBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_pieces', 'batch_token_lens', 'batch_token_num', 'batch_text', 'batch_triggers', 'batch_questions', 'batch_labels']
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
        batch_questions=[instance["question"] for instance in batch], 
        batch_labels=[instance["label"] for instance in batch],
    )

class MQAEETriClsTrainer(object):
    def __init__(self, config, type_set=None):
        self.tokenizer = None
        self.model = None
        self.config = config
        self.type_set = type_set
        if self.type_set:
            self.label_list = [x for x in sorted(self.type_set["trigger"])]
    
    def load_model(self, checkpoint=None):
        
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(os.path.join(checkpoint, "best_model.state"), map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            
            self.model = MQAEETriClfModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('roberta-'):
                self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model_path, cache_dir=self.config.cache_dir, do_lower_case=False)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_path, cache_dir=self.config.cache_dir, do_lower_case=False)
            special_tokens = ["<pos>", "</pos>"]
            logger.info(f"Add tokens {special_tokens}")
            self.tokenizer.add_tokens(special_tokens)
            self.model = MQAEETriClfModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        question = "The trigger word is [trigger] <pos>[trigger_position]</pos>, [event_type], [argument_roles]?"
        new_data = []
        for dt in data:
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
            pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            
            for trigger in dt["triggers"]:
                
                start, end, event_type, text_span = trigger[0], trigger[1], trigger[2], trigger[3]
                new_question = question.replace("[trigger]", text_span).replace("[trigger_position]", str(start))
                
                arg_roles = ', '.join(sorted(patterns[self.config.dataset][event_type]))
                pos_question = new_question.replace("[event_type]", event_type).replace("[argument_roles]", arg_roles)
                pos_question = pos_question + " [SEP] " + dt["text"]
                
                new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "pieces": [p for w in pieces for p in w], 
                          "token_lens": token_lens, 
                          "token_num": len(dt["tokens"]), 
                          "text": dt["text"], 
                          "triggers": dt["triggers"],
                          "question": pos_question,
                          "label": 1
                         }

                new_data.append(new_dt)
                
                label_list_ = [x for x in self.label_list if x != event_type]
                np.random.shuffle(label_list_)
                label_list_ = label_list_[:self.config.max_sample_neg]
                for neg_event_type in label_list_:
                    if neg_event_type == event_type:
                        continue
                
                    arg_roles = ', '.join(sorted(patterns[self.config.dataset][neg_event_type]))
                    neg_question = new_question.replace("[event_type]", neg_event_type).replace("[argument_roles]", arg_roles)
                    neg_question = neg_question + "[SEP]" + dt["text"]
                    
                    new_dt = {"doc_id": dt["doc_id"], 
                          "wnd_id": dt["wnd_id"], 
                          "tokens": dt["tokens"], 
                          "pieces": [p for w in pieces for p in w], 
                          "token_lens": token_lens, 
                          "token_num": len(dt["tokens"]), 
                          "text": dt["text"], 
                          "triggers": dt["triggers"],
                          "question": neg_question,
                          "label": 0
                         }

                    new_data.append(new_dt)
                
        return new_data
    
    def process_eval_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"
        
        logger.info("Removing over-length examples")
        
        new_data = []
        for dt in data:
            
            if len(dt["tokens"]) > self.config.max_length:
                continue
            
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
                      "question": None,
                      "label": None
                     }
            
            new_data.append(new_dt)
                
        return new_data

    def train(self, train_data, dev_data):
        self.load_model()
        internal_train_data = self.process_data(train_data)
        internal_dev_data = self.process_eval_data(dev_data)

        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if n.startswith('base_model')],
                'lr': self.config.base_model_learning_rate, 'weight_decay': self.config.base_model_weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not n.startswith('base_model')],
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay
            },
        ]
        train_batch_num = len(internal_train_data) // self.config.train_batch_size + (len(internal_train_data) % self.config.train_batch_size != 0)
        optimizer = Adam(params=param_groups)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                                    num_training_steps=train_batch_num*self.config.max_epoch)
                
        best_scores = {"trigger_cls": {"f1": 0.0}}
        best_epoch = -1
        
        for epoch in range(1, self.config.max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")
            
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))
            
            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            for batch_idx, batch in enumerate(DataLoader(internal_train_data, batch_size=self.config.train_batch_size // self.config.accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=ED_collate_fn)):
                
                loss = self.model(batch)
                
                if loss:
                    loss = loss * (1 / self.config.accumulate_step)
                    cummulate_loss.append(loss.item())
                    loss.backward()
                
                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
            progress.close()
            logger.info(f"Average training loss: {np.mean(cummulate_loss)}")

            predictions = self.internal_predict(internal_dev_data, split="Dev")
            dev_scores = compute_ED_scores(predictions, dev_data, metrics={"trigger_id", "trigger_cls"})

            print(f"Dev {epoch}")
            print_scores(dev_scores)
            
            if dev_scores["trigger_cls"]["f1"] >= best_scores["trigger_cls"]["f1"]:
                logger.info("Saving best model")
                state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_scores = dev_scores
                best_epoch = epoch
                
            logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
            logger.info(pprint.pformat({"best_epoch": best_epoch, "best_scores": best_scores}))
        
    def internal_predict(self, eval_data, split="Dev"):
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, collate_fn=ED_collate_fn)):
            progress.update(1)
            batch_pred_triggers = self.model.predict(batch, batch.batch_triggers)
            for doc_id, wnd_id, tokens, text, pred_triggers in zip(batch.batch_doc_id, batch.batch_wnd_id, batch.batch_tokens, 
                                                                   batch.batch_text, batch_pred_triggers):
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "triggers": pred_triggers
                             }
                
                predictions.append(prediction)
        progress.close()
        
        return predictions
    
    def predict(self, data):
        assert self.tokenizer and self.model
        internal_data = self.process_eval_data(data)
        predictions = self.internal_predict(internal_data, split="Test")
        return predictions