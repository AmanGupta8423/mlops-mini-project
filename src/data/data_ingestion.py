import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str) -> dict:
    'Load Parameters from yaml'
    try:
        with open(params_path,'r') as file:
            param = yaml.safe_load(file)
            logger.debug('Parameters retreval from %s',params_path)
            return param
    except FileNotFoundError:
        logger.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise

def load_data(data_url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded Sucessfully')
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
        raise

def processing_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns = ['tweet_id'],inplace = True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({
            'happiness':1,
            'sadness':0
        }).astype('int64')
        logger.debug('Data Processing Completed')
        return final_df
    except KeyError as e:
        logger.error('Missing column in the DataFrame: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error during the preprocessing: %s',e)
        raise

def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path:str):
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index = False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and Test data saved to %s',raw_data_path)
        
    except Exception as e:
        logger.error('Unexpected Error occurred while saving the data: %s',e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url= 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = processing_data(df)
        train_df, test_df = train_test_split(final_df, test_size=test_size,random_state = 42)
        save_data(train_data = train_df,test_data=test_df, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        raise


if __name__ == '__main__':
    main()