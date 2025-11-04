import json 
from pathlib import Path 
import math 
from PIL import Image 
from datasets import Dataset, DatasetDict
import datetime 


def prepend_sample(data_path, ext, prefix='sample_'): 
    """f is a Python Path object with attributes parent & name, and method rename; 
    renames a file in place with a prefix prepended to the previous file name"""

    for f in data_path.glob(f'*{ext}'): 
        if f.is_file():
            # find the file name in the Path object f  
            # f.rename 
            new_file_name = f.parent / Path(prefix + f.name)
            f.rename(new_file_name)


def generate_hf_dataset(img_path, label_path, prompt):

    data = []
    sample_keys = ['id', 'image', 'prompt', 'target']

    for f in label_path.glob('*.json'):

        sample_dict = {key: None for key in sample_keys}

        if f.is_file():
            with open(f, 'r', encoding='utf-8') as record:
                sample = json.load(record)

            sample_dict['id'] = sample['figure_id']
            sample_dict['prompt'] = prompt

            # load the image in PIL and insert into dict
            sample_path = img_path / Path('sample_' + sample['figure_id'] + '.png')
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

            sample_dict['target'] = mat_list

            data.append(sample_dict)

    return data 


def generate_jsonl_manifest(img_path, label_path, prompt, split='unknown', output_file=None): 
    data = []
    sample_keys = ['id', 'image_path', 'prompt', 'target']

    if output_file is None:
        output_file = 'data/rheo_sigmoid/manifests/' + split + '_manifest_' + str(datetime.date.today()) + '.jsonl'

    for f in label_path.glob('*.json'):

        sample_dict = {key: None for key in sample_keys}

        if f.is_file():
            with open(f, 'r', encoding='utf-8') as record:
                sample = json.load(record)

            sample_dict['id'] = sample['figure_id']
            sample_dict['prompt'] = prompt
            sample_dict['image_path'] = img_path / Path('sample_' + sample['figure_id'])

            sample_dict['image_path'] = sample_dict['image_path'].as_posix()
 
            # loop through the 'materials' list and add list elements to target
            mat_list = []
            for i in range(len(sample['materials'])):

                material_dict = {"label": sample['materials'][i]['label_in_legend'], 
                                "Gp_Pa": math.floor(sample['materials'][i]['Gp_plateau_Pa']), 
                                "tau_y_Pa": math.floor(sample['materials'][i]['tau_y_Pa']), 
                                "tau_f_Pa": math.floor(sample['materials'][i]['tau_f_Pa'])
                                }
                                    
                mat_list.append(material_dict)

            # serialize the list of dicts from material loop
            # sample_dict['target'] = json.dumps(mat_list)
            sample_dict['target'] = mat_list

            data.append(sample_dict)

    # serialize the entire list of dictionaries and write to output file as json lines
    with open(output_file, 'w') as manifest:
        for item in data: 
            json_line = json.dumps(item)
            manifest.write(json_line + '\n')


if __name__ == "__main__": 

    # prepend_sample(data_path=Path('data/rheo_sigmoid/test/images'), ext='.png')

    prompt = ["You are a rheology assistant. What are the storage modulus, yield stress, and flow stress for each material in the rheology plot? Extract the rheological parameters and respond strictly in json."]

    train_data = Dataset.from_list(generate_hf_dataset(img_path=Path('data/rheo_sigmoid/train/images'),
                                                      label_path=Path('data/rheo_sigmoid/train/labels'),
                                                        prompt=prompt))

    train_manifest = generate_jsonl_manifest(img_path=Path('data/rheo_sigmoid/train/images'),
                                            label_path=Path('data/rheo_sigmoid/train/labels'),
                                              prompt=prompt,
                                                split='train')
      
    val_data = Dataset.from_list(generate_hf_dataset(img_path=Path('data/rheo_sigmoid/val/images'),
                                                      label_path=Path('data/rheo_sigmoid/val/labels'),
                                                        prompt=prompt))

    val_manifest = generate_jsonl_manifest(img_path=Path('data/rheo_sigmoid/val/images'),
                                            label_path=Path('data/rheo_sigmoid/val/labels'),
                                              prompt=prompt,
                                                split='validation')

    test_data = Dataset.from_list(generate_hf_dataset(img_path=Path('data/rheo_sigmoid/val/images'),
                                                      label_path=Path('data/rheo_sigmoid/val/labels'),
                                                        prompt=prompt))

    test_manifest = generate_jsonl_manifest(img_path=Path('data/rheo_sigmoid/test/images'),
                                            label_path=Path('data/rheo_sigmoid/test/labels'),
                                              prompt=prompt,
                                                split='test')

    hf_data = DatasetDict({'train': train_data, 
                        'validation': val_data,
                        'test': test_data})

    hf_data.push_to_hub('dchip95/rheology_dataset_pixels_20251103')