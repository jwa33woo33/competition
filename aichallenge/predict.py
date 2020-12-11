import os
import sys
import json
import glob
import subprocess
from collections import defaultdict
from PIL import Image


argv = sys.argv[1]
internal = False


def create_testdev():
    """
    Create testdev json file
    """
    print('Generating test-dev.json...')
    with open('./result_jsons/category.json', 'r') as json_file:
        category_dict = json.load(json_file)

    image_files = sorted(list(glob.glob(f'{argv}/*.jpg')))
    if len(image_files) == 0:
        raise ValueError('폴더에 사진이 하나도 없어서 COCO 포맷 test-dev를 생성할 수 없슴!')
    else:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    test_dev = {
        'images': [],
        "categories": category_dict['categories']}

    for i, img_file in enumerate(image_files):
        try:
            img = Image.open(img_file)
            w, h = img.size
            test_dev['images'].append({
                "file_name": os.path.basename(img_file),
                "height": h,
                "width": w,
                "id": i})
        except:
            print(img_file)

    print(len(image_files), len(test_dev['images']))

    # Save test dev json file
    print('Save json...')
    # os.path.isfile()
    with open('./result_jsons/test_dev.json', 'w', encoding='utf-8') as f:
        json.dump(test_dev, f, indent='\t')


def create_testdev_internal():
    """
    Create testdev json file
    """
    print('Generating test-dev.json...')
    with open('./result_jsons/category.json', 'r') as json_file:
        category_dict = json.load(json_file)
        
    with open('./result_jsons/valid.json', 'r') as f:
        valid = json.load(f)

    ssibal = []
    for image_info in valid['images']:
        path = image_info['file_name'].split('/')[1]
        ssibal.append({
            'file_name': path,
            'id': image_info['id'],
            'height': image_info['height'],
            'width': image_info['width']
        })

    test_dev = {
        'images': ssibal,
        "categories": category_dict['categories']}

    # Save test dev json file
    print('Save json...')
    # os.path.isfile()
    with open('./result_jsons/test_dev.json', 'w', encoding='utf-8') as f:
        json.dump(test_dev, f, indent='\t')


def run_predict_model(shell=True, check=True):
    """
    
    """
    try:
        import effdet
    except ImportError:
        print('Install effdet...')
        subprocess.run(['python setup.py install'], shell=shell, check=check)

    print('Run model...')
    pretrained_weight = './output/train/model_d3/checkpoint-502.pth.tar'
    model_name = 'tf_efficientdet_d3'
    subprocess.run([f'python ./validate.py {argv} --model={model_name}'
        + f' --split=testdev --num-classes=7 -b=8 --native-amp' 
        + f' --pretrained --checkpoint={pretrained_weight}'], shell=shell, check=check)


def create_submission_file(threshold=0):
    """
    Arguments:
        results.json: model output
    """

    with open('./result_jsons/test_dev.json', 'r', encoding='utf-8') as json_file:
        test_dev = json.load(json_file)

    with open('./result_jsons/temp.json', 'r', encoding='utf-8') as json_file:
        model_outputs = json.load(json_file)

    classes = {cat['id']: cat['name'] for cat in test_dev['categories']}
    file_names = {image['id']: image['file_name'] for image in test_dev['images']}

    converted_results = defaultdict(list)
    for output in model_outputs:
        converted_results[output['image_id']].append({
            'bbox': output['bbox'],
            'score': output['score'],
            'category_id': output['category_id']
        })

    converted_cnt = 0
    for key, values in converted_results.items():
        for value in values:
            converted_cnt += 1

    testdev_list = [image['id'] for image in test_dev['images']]
    results = defaultdict(list)
    for testdev_id in testdev_list:
        annots = converted_results.get(testdev_id, {
            'bbox': None, 'score': 0.0, 'category_id': ''})

        results[testdev_id] += (annots)

    check_cnt = 0
    for key, values in results.items():
        for value in values:
            check_cnt += 1


    print('Generating submission file...')
    submission = {'annotations': []}
    for key, values in results.items():
        temp = {
            'id': key,
            'file_name': file_names[key],
            'object': []}

        for value in values:
            if value['bbox'] is not None:
                if value['score'] > threshold:
                    temp['object'].append({
                        'box': value['bbox'],
                        'label': classes[value['category_id']]})
                    

        if len(temp['object']) == 0:
            temp['object'].append({'box': [], 'label': ''})
            

        submission['annotations'].append(temp)
    
    submit_cnt = 0
    for annot in submission['annotations']:
        for ann in annot['object']:
            submit_cnt += 1

    print('Check points...')
    print(f'- Converted: {len(converted_results)}({converted_cnt})')
    print(f'- Resulted:  {len(results)}({check_cnt})')
    print(f'- Final sub: {len(submission["annotations"])}({submit_cnt})')

    # Save submission json file
    print('Save t3_res_U0000000225.json...')
    with open('./t3_res_U0000000225.json', 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent='\t')


def validation():
    print('Check submission file...')
    with open('./t3_res_U0000000225.json', 'r') as json_file:
        data = json.load(json_file)

    try:
        annots = data['annotations']
        if len(annots) == 0:
            print('\n예측하나도 안됨! ㅇㅅㅇ\n')
            raise ValueError

        for annot in annots:
            _ = annot['file_name']

    except KeyError as e:
        print('에러!', e)


if __name__=='__main__':
    if internal:
        create_testdev_internal()
    else:
        create_testdev()

    run_predict_model(shell=True, check=True)
    create_submission_file(threshold=0.5)
    validation()

