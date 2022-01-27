#!/usr/bin/env python
"""
This is the data train-test splitting module.
"""

import argparse
import logging
import wandb

import pandas as pd
import time

from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    Split the data into train set and test set.
    """

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    artifact = run.use_artifact(args.input)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)
    logger.info('Artifact {} loaded sucessful at {}'.format(artifact_path, time.strftime('%b_%d_%Y_%H_%M_%S')))

    train_data, test_data = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    train_data_path = './trainval_data.csv' 
    test_data_path = './test_data.csv'

    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)
    logger.info("Save the data sets in the current dir sucessful.")

    artifact = wandb.Artifact(
        name='trainval_data.csv',
        type='train_val dataset',
        description='train set'
    )

    artifact.add_file(train_data_path)
    run.log_artifact(artifact)
    artifact.wait()
    logger.info('Train data set upload to wandb sucessful.')

    artifact = wandb.Artifact(
        name='test_data.csv',
        type='test dataset',
        description='test set'
    )

    artifact.add_file(test_data_path)
    run.log_artifact(artifact)
    artifact.wait()
    logger.info('Test data set upload to wandb sucessful.')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="These steps split the data into train and test sets")

    parser.add_argument(
        "--input", 
        type=str,
        help="The input artifact name.",
        required=True
    )

    parser.add_argument(
        "--test_size", 
        type=float,
        help="The ratio of the train and test sets.",
        required=True
    )

    parser.add_argument(
        "--random_seed", 
        type=int,
        help="A random seed making sure that the result is reproduciable",
        default=42
    )

    parser.add_argument(
        "--stratify_by", 
        type=str,
        help="Column to use for stratification (if any)",
        default='none'
    )

    args = parser.parse_args()

    go(args)