import os, logging, json, random, datetime, pprint
import numpy as np
import torch
from argparse import Namespace
from TriIdtrainer import MQAEETriIdTrainer
from TriClstrainer import MQAEETriClsTrainer

logger = logging.getLogger(__name__)

VALID_TASKS = ["TriId", "TriCls", "EAE"]

TRAINER_MAP = {
    ("MQAEE", "TriId"): MQAEETriIdTrainer,  
    ("MQAEE", "TriCls"): MQAEETriClsTrainer,
    ("MQAEE", "EAE"): MQAEETriIdTrainer, 
}

def load_config(config_fn):
    with open(config_fn, 'r') as f:
        config = json.load(f)
    config = Namespace(**config)
    assert config.task in VALID_TASKS, f"Task must be in {VALID_TASKS}"

    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def set_gpu(gpu_device):
    if gpu_device >= 0:
        torch.cuda.set_device(gpu_device)

def set_logger(config):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
    output_dir = os.path.join(config.output_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = os.path.join(output_dir, "train.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                        handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
    logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
    
    # save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
        json.dump(vars(config), fp, indent=4)
        
    config.output_dir = output_dir
    config.log_path = log_path
    
    return config

def load_all_data(config):
    if config.task == "TriId" or config.task == "TriCls":
        train_data, train_type_set = load_ED_data(config.train_file)
        dev_data, dev_type_set = load_ED_data(config.dev_file)
        test_data, test_type_set = load_ED_data(config.test_file)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"]}
        logger.info("There are {} trigger types in total".format(len(type_set["trigger"])))
    elif config.task == "EAE":
        train_data, train_type_set = load_EAE_data(config.train_file)
        dev_data, dev_type_set = load_EAE_data(config.dev_file)
        test_data, test_type_set = load_EAE_data(config.test_file)
        type_set = {"trigger": train_type_set["trigger"] | dev_type_set["trigger"] | test_type_set["trigger"], 
                    "role": train_type_set["role"] | dev_type_set["role"] | test_type_set["role"]}
        logger.info("There are {} trigger types and {} role types in total".format(len(type_set["trigger"]), len(type_set["role"])))
    else:
        raise ValueError(f"Task {config.task} is not supported")
    
    return train_data, dev_data, test_data, type_set

def load_ED_data(file):

    with open(file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    instances = []
    for dt in data:
        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        triggers = []
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            triggers.append(trigger)

        triggers.sort(key=lambda x: (x[0], x[1]))
        
        instance = {"doc_id": dt["doc_id"], 
                    "wnd_id": dt["wnd_id"], 
                    "tokens": dt["tokens"], 
                    "text": dt["text"], 
                    "triggers": triggers,
                   }

        instances.append(instance)

    trigger_type_set = set()
    for instance in instances:
        for trigger in instance['triggers']:
            trigger_type_set.add(trigger[2])

    type_set = {"trigger": trigger_type_set}
    
    logger.info('Loaded {} ED instances ({} trigger types) from {}'.format(
        len(instances), len(trigger_type_set), file))
    
    return instances, type_set

def load_EAE_data(file):

    with open(file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    instances = []
    for dt in data:
        
        entities = dt['entity_mentions']

        event_mentions = dt['event_mentions']
        event_mentions.sort(key=lambda x: x['trigger']['start'])

        entity_map = {entity['id']: entity for entity in entities}
        for i, event_mention in enumerate(event_mentions):
            # trigger = (start index, end index, event type, text span)
            trigger = (event_mention['trigger']['start'], 
                       event_mention['trigger']['end'], 
                       event_mention['event_type'], 
                       event_mention['trigger']['text'])

            arguments = []
            for arg in event_mention['arguments']:
                mapped_entity = entity_map[arg['entity_id']]
                
                # argument = (start index, end index, role type, text span)
                argument = (mapped_entity['start'], mapped_entity['end'], arg['role'], arg['text'])
                arguments.append(argument)

            arguments.sort(key=lambda x: (x[0], x[1]))
            
            instance = {"doc_id": dt["doc_id"], 
                        "wnd_id": dt["wnd_id"], 
                        "tokens": dt["tokens"], 
                        "text": dt["text"], 
                        "trigger": trigger, 
                        "arguments": arguments, 
                       }

            instances.append(instance)
            
    trigger_type_set = set()
    for instance in instances:
        trigger_type_set.add(instance['trigger'][2])

    role_type_set = set()
    for instance in instances:
        for argument in instance["arguments"]:
            role_type_set.add(argument[2])
                
    type_set = {"trigger": trigger_type_set, "role": role_type_set}
    
    logger.info('Loaded {} EAE instances ({} trigger types and {} role types) from {}'.format(
        len(instances), len(trigger_type_set), len(role_type_set), file))
    
    return instances, type_set