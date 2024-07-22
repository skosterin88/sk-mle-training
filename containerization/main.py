#!/usr/bin/env python

import requests
import json
import yaml
from pathlib import Path

from yolo_call import query



def main():

    config = yaml.safe_load(Path('config.yaml').read_text())

    model_config = config['model_params']
    input_config = config['input_params']
    output_config = config['output_params']

    model_url = model_config['url']
    hf_access_token = model_config['hf_access_token']
    headers = model_config['headers']
    headers['Authorization'] = headers['Authorization']\
    .format(hf_access_token=hf_access_token)
    
    input_filename = input_config['default_image']
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
    
