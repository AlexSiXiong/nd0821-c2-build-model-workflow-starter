#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb

import pandas as pd
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Cleaning the NYC data downloaded.

    Note: 
        the missing value processing works will be conducted in the inference pipeline.

    There are a few steps:
    1. after talking to the stakeholders, I remove all the records with abnormal prices.
    2. convert the 'last_review' col to date time.
    3. filter all the records in the New York city.
    4. save the processed file locally.
    5. upload the artifact.
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()


    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)
    logger.info('Artifact {}loaded sucessful at {}'.format(artifact_path, time.strftime('%b_%d_%Y_%H_%M_%S')))

    # 1. after talking to the stakeholders, I remove all the records with abnormal prices.
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # 2. convert the 'last_review' col to date time.
    df['last_review'] = pd.to_datetime(df['last_review'])

    # 3. filter all the records in the New York city.
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info(f"Complete the basic data cleaning works sucessful.")

    # 4. save the processed file
    df.to_csv('./basic_cleaning_data_{}.csv'.format(time.strftime('%Y_%m_%d_%H')))
    logger.info(f"Save the basic data cleaning df in the current dir sucessful.", index=False)

    # 5. upload the artifact
    # this prat is new 
    artifact= wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file('basic_clean_data.csv')
    
    run.log_artifact(artifact)
    artifact.wait()

    logger.info('Basic cleaned data upload to wandb sucessful.')

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The input artifact name.",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The output artifact name.",
        required=True
    )


    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="The description of the artifact output",
        required=True
    )


    parser.add_argument(
        "--min_price", 
        type=float,
        help='The min price for filtering the data. Please type a float.',
        default=10
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='The max price for filtering the data. Please type a float.',
        default=350,
    )


    args = parser.parse_args()

    go(args)
