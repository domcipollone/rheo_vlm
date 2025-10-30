import json 
from pathlib import Path 
import math 
from PIL import Image 
from datasets import Dataset, DatasetDict


def generate_jsonl_file(img_path, label_path, prompt):

    data = []
    sample_keys = ['id', 'image', 'prompt', 'target']

    for f in label_path.glob('*.json'):

        sample_dict = {key: None for key in sample_keys}

        if f.is_file():
            with open(f, 'r', encoding='utf-8') as record:
                sample = json.loads(record)

                sample_dict['id'] = sample['figure_id']
                sample_dict['prompt'] = (sample['prompt'])

                # load the image in PIL and insert into dict
                sample_path = img_path / Path('sample_' + sample['figure_id'])
                sample_dict['image'] = Image.open(sample_path).convert("RGB")

                # loop through the 'materials' list and add list elements to target
                mat_list = []
                for i in range(len(sample['materials'])):

                    material_dict = {'label': sample['materials'][i]['label_in_legend'], 
                                     'Gp_Pa': math.floor(sample['materials'][i]['Gp_plateau_Pa']), 
                                     'tau_y_Pa': math.floor(sample['materials'][i]['tau_y_Pa']), 
                                     'tau_f_Pa': math.floor(sample['materials'][i]['tau_f_Pa'])
                                     }
                                     
                    mat_list.append(material_dict)

                # serialize the list of dicts 
                sample_dict['target'] = json.dumps(mat_list)

                data.append(sample_dict)

    return data 