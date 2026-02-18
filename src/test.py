from unsloth import FastLanguageModel
from utils_io import setup_config, save_json, get_current_time_string
from runner import Runner
from calculate_metrics import RelationExtractionEvaluator
import argparse
import os
import yaml
import pandas as pd


def main(args):
    config = setup_config(args)

    # Load model (no training setup)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_length"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=config["load_in_8bit"],
        full_finetuning=False,
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure chat template
    if tokenizer.chat_template is None and hasattr(model.config, "model_type"):
        if "Qwen3" in config["model_name"]:
            tokenizer.chat_template = open("./model_info/qwen3.jinja").read()
        else:
            chat_template_dict = yaml.safe_load(
                open("./model_info/chat_templates.yaml")
            )
            tokenizer.chat_template = chat_template_dict.get(
                model.config.model_type
            )
        print(
            f'Chat template not found, using the one for model type "{model.config.model_type}"'
        )

    evaluator = RelationExtractionEvaluator(mode="EE")

    runner = Runner(
        model=model,
        tokenizer=tokenizer,
        config=config,
        evaluator=evaluator,
    )

    # Load data
    df_train = pd.read_json(
        os.path.join(config["dataset_path"], "train.json")
    )
    df_test = pd.read_json(
        os.path.join(config["dataset_path"], "test.json")
    )

    if config["eval_samples"]:
        df_test = df_test[: config["eval_samples"]]

    # Run evaluation only
    test_results = runner.evaluate(df_test, df_train, split="test")
    test_results['config'] = config
    print(f"Test results: {test_results}")
    save_path = os.path.join(config["results_dir"], f"test_results_{get_current_time_string()}.json")
    print(f'Saving results to: {save_path}')
    save_json(
        test_results,
        save_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned language model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ade",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--n_icl_samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dtype_str",
        type=str,
        default="bfloat16",
    )
    parser.add_argument(
        "--entitytypes",
        default="entity2type.json",
    )
    parser.add_argument(
        "--prompt_filename",
        default="default.yaml",
    )
    parser.add_argument(
        "--desc_schema",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_results",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/.testing",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--load_in_8bit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--verbose_preds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--verbose_metrics",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
    )
    parser.add_argument(
        "--do_train",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--rationale",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_prompt",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    main(args)
