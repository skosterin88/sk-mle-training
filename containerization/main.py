#!/usr/bin/env python

import sys
import requests
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

from yolo_call import query



def main():

    load_dotenv()
    
    hf_access_token = os.getenv('hf_access_token')
    
    config = yaml.safe_load(Path('config.yaml').read_text())

    model_config = config['model_params']
    input_config = config['input_params']
    output_config = config['output_params']

    model_url = model_config['url']
    # hf_access_token = model_config['hf_access_token']
    headers = model_config['headers']
    headers['Authorization'] = headers['Authorization']\
    .format(hf_access_token=hf_access_token)
    
    args = sys.argv
    
    default_image = input_config['default_image']
    
    if len(args) > 1:
    
    	input_filename = args[1]
    
    else:
    
    	input_filename = default_image
    	
    output_filename = output_config['output_file_name']

    output_dict = {}

    output_list = query(input_filename, model_url, headers)
    output_dict[input_filename] = output_list

    with open(output_filename, 'w', encoding='utf-8') as f:

       json.dump(output_dict, f, ensure_ascii=False)
        
    print(f'Output saved to {output_filename}!')
    print('Results of model usage:')
    print(output_dict)


if __name__ == '__main__':

    main()
    
