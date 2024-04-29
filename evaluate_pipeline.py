import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, set_seed, set_gpu, load_data, convert_TriId_to_TriCls, convert_TriCls_to_ArgExt, combine_ED_and_EAE_to_E2E
from scorer import compute_scores, print_scores

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--data', default="./data/test.json")
    parser.add_argument('--triid_model', default="./outputs/MQAEE_TriId_ace05_bert-large-cased")
    parser.add_argument('--tricls_model', default="./outputs/MQAEE_TriCls_ace05_bert-large-cased")
    parser.add_argument('--argext_model', default="./outputs/MQAEE_ArgExt_ace05_bert-large-cased")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    set_seed(args.seed)
    set_gpu(args.gpu)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    
    assert args.triid_model and args.tricls_model and args.argext_model
    # load data
    triid_eval_data, _ = load_data("TriId", args.data)
    gold_data, _ = load_data("E2E", args.data)

    # load TriId trainer and model
    triid_config = load_config(os.path.join(args.triid_model, "config.json"))
    triid_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(triid_config), indent=4)}")
    triid_trainer_class = TRAINER_MAP[(triid_config.model_type, triid_config.task)]
    triid_trainer = triid_trainer_class(triid_config)
    triid_trainer.load_model(checkpoint=args.triid_model)
    triid_predictions = triid_trainer.predict(triid_eval_data)
    scores = compute_scores(triid_predictions, triid_eval_data, "TriId")
    print_scores(scores)
    
    tricls_eval_data = convert_TriId_to_TriCls(triid_predictions, triid_eval_data)
    
    # load TriCls trainer and model
    tricls_config = load_config(os.path.join(args.tricls_model, "config.json"))
    tricls_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(tricls_config), indent=4)}")
    tricls_trainer_class = TRAINER_MAP[(tricls_config.model_type, tricls_config.task)]
    tricls_trainer = tricls_trainer_class(tricls_config)
    tricls_trainer.load_model(checkpoint=args.tricls_model)
    tricls_predictions = tricls_trainer.predict(tricls_eval_data)

    argext_eval_data = convert_TriCls_to_ArgExt(tricls_predictions, tricls_eval_data)

    # load ArgExt trainer and model
    argext_config = load_config(os.path.join(args.argext_model, "config.json"))
    argext_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(argext_config), indent=4)}")
    argext_trainer_class = TRAINER_MAP[(argext_config.model_type, argext_config.task)]
    argext_trainer = argext_trainer_class(argext_config)
    argext_trainer.load_model(checkpoint=args.argext_model)
    argext_predictions = argext_trainer.predict(argext_eval_data)

    e2e_predictions = combine_ED_and_EAE_to_E2E(tricls_predictions, argext_predictions)
    
    scores = compute_scores(e2e_predictions, gold_data, "E2E")
    print("Evaluate")
    print_scores(scores)
        
if __name__ == "__main__":
    main()