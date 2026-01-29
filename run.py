import sys
from logging import getLogger
import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_flops
from trainer import LanguageLossTrainer, SelectedUserTrainer, ITEMLanguageLossTrainer
from utils import get_model
from dataset import BPRDataset, ITEMBPRDataset
import openai
import os
os.environ["api_base"] = "https://api.holdai.top/v1"
# os.environ["api_base"] = "https://lonlie.plus7.plus/v1"
openai.api_base = os.environ.get("api_base")



def run_baseline(model_name, dataset_name, **kwargs):
    props = ['props/overall.yaml', f'props/{model_name}_{dataset_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)

    model_class = get_model(model_name)

    # configurations initialization 
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    if model_name == 'BPR' or model_name == 'UUPretrain' or model_name == 'ReRec' or model_name == 'AllReRec' or model_name == 'IITest' or model_name == 'TestGames' or model_name == 'SparseReRec'\
            or model_name in ['TestPantry', 'TestOffice', 'IITestDiag', 'IITestDiagNew', 'TestOfficeBPR', 'TestOfficeUUPretrain','UserReRec','AgentCF']:
        dataset = BPRDataset(config)
    elif model_name in ['UUTest','UUTestDiag','TestOfficeUUTest']:
        dataset = ITEMBPRDataset(config)
    else:
        dataset = create_dataset(config)

    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    if model_name in ['SASRec','BPRMF']:
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    elif model_name in ['UUTest', 'UUTestDiag','TestOfficeUUTest','TestOfficeUUTestDiag']:
        trainer = ITEMLanguageLossTrainer(config,model,dataset)
    else:
        trainer = LanguageLossTrainer(config,model,dataset)

    if not config['test_only']:
        # # model training
        trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])

    # model evaluation
    test_result = trainer.evaluate(test_data, model_file='./AgentCF-Sep-07-2024_16-09-29.pth', load_best_model=False, show_progress=config["show_progress"])
    print(test_result)
    diver_results = model.print_diversity_results()

    # logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    logger.info(set_color("diversity result", "yellow") + f": {diver_results}")

    return model_name, dataset_name, {
        # "best_valid_score": best_valid_score,
        # "valid_score_bigger": config["valid_metric_bigger"],
        # "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="STEAM", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="cd", help="name of datasets")
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--api_batch", type=int, default=20)
    parser.add_argument("--test_only", type=bool, default=False)

    args, _ = parser.parse_known_args()

    run_baseline(args.model, args.dataset)
