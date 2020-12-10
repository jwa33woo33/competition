import os
import json
from collections import defaultdict


def convert():
    with open('./annotations.json') as json_file:
        valid_data = json.load(json_file)


    id_filename_map = {cat['id']: os.path.basename(cat['file_name']) for cat in valid_data['images']}
    classes_map = {cat['id']: cat['name'] for cat in valid_data['categories']}

    merged_val = defaultdict(list)
    for val in valid_data['annotations']:
        merged_val[val['image_id']].append((
            val['category_id'], val['bbox']))


    groud_truth = {'annotations': []}
    for img_id, infos in merged_val.items():
        single_img = {
            'id': img_id,
            'file_name': id_filename_map[img_id],
            'object': []}
        
        for info in infos:
            single_img['object'].append({
                'box': info[1],
                'label': classes_map[info[0]]})

        groud_truth['annotations'].append(single_img)


    with open('./ground_truth.json', 'w') as f:
        json.dump(groud_truth, f, indent='\t')



convert()

