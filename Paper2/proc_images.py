categories = []

categories = ({"id": 6508809, "name": "bike"},
                  {"id": 6508801, "name": "bus"},
                  {"id": 6508800, "name": "car"},
                  {"id": 6508802, "name": "drivable area"},
                  {"id": 6508803, "name": "lane"},
                  {"id": 6508810, "name": "motor"},
                  {"id": 6508806, "name": "person"},
                  {"id": 6508808, "name": "rider"},
                  {"id": 6508807, "name": "traffic light"},
                  {"id": 6508804, "name": "traffic sign"},
                  {"id": 6508811, "name": "train"},
                  {"id": 6508805, "name": "truck"},)


categories2 = ({"id": 0, "name": "bike"},
                  {"id": 1, "name": "bus"},
                  {"id": 2, "name": "car"},
                  {"id": 3, "name": "drivable area"},
                  {"id": 4, "name": "lane"},
                  {"id": 5, "name": "motor"},
                  {"id": 6, "name": "person"},
                  {"id": 7, "name": "rider"},
                  {"id": 8, "name": "traffic light"},
                  {"id": 9, "name": "traffic sign"},
                  {"id": 10, "name": "train"},
                  {"id": 11, "name": "truck"},)

import json
import os

# replace every catorie id with the new one
def replace_categories(categories, categories2, path):
    # ouvre le fichier
    with open(path, 'r') as f:
        data = json.load(f)
        for i in range(len(data['annotations'])):
            for j in range(len(categories)):
                if data['annotations'][i]['category_id'] == categories[j]['id']:
                    data['annotations'][i]['category_id'] = categories2[j]['id']
        data['categories'] = categories2
        f.close()
    path = path[:-5] + '2.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.close()


images = []
annotations = []
image_id = 1
annotation_id = 1
path = "dataset/bdd100k-DatasetNinja/train/ann"

def process_json_file(filename, images, annotations, annotation_id, image_id, path):

  with open(path+'/'+filename, 'r') as f:
    data = json.load(f)
    #name = f.name[:-5]
    name = filename[:-5]

    images.append({
              "id": image_id,
              "file_name": f"{name}",  # Vous pouvez adapter cela en fonction de vos noms de fichiers réels
              "height": data['size']['height'],
              "width": data['size']['width']
          })
  for obj in data['objects']:

      if(obj['geometryType']=="rectangle"): # only bbox
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "bbox": [obj['points']['exterior'][0][0], obj['points']['exterior'][0][1],
                    obj['points']['exterior'][1][0],
                    obj['points']['exterior'][1][1]],
            "category_id": obj['classId']
        })
        annotation_id += 1
  #image_id += 1


for filename in os.listdir(path):
    if filename.endswith('.json'):
        process_json_file(filename, images, annotations, annotation_id, image_id,path)
        image_id += 1

output_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open('input/train2.json', 'w') as f:
    json.dump(output_data, f, indent=2)
    f.close()
print("Train json done")


path = 'dataset/bdd100k-DatasetNinja/test/ann'
images = []
annotations = []
image_id = 1
annotation_id = 1

for filename in os.listdir(path):
    if filename.endswith('.json'):
        process_json_file(filename, images, annotations, annotation_id, image_id,path)
        image_id += 1

output_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open('input/test.json', 'w') as f:
    json.dump(output_data, f, indent=2)
    f.close()

print("Test json done")
path = 'dataset/bdd100k-DatasetNinja/val/ann'
images = []
annotations = []
image_id = 1
annotation_id = 1

for filename in os.listdir(path):
    if filename.endswith('.json'):
        process_json_file(filename, images, annotations, annotation_id, image_id,path)
        image_id += 1


output_data = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}


with open('input/val.json', 'w') as f:
    json.dump(output_data, f, indent=2)
    f.close()
print("Val json done")


replace_categories(categories, categories2, "input/train.json")
replace_categories(categories, categories2, "input/test.json")
replace_categories(categories, categories2, "input/val.json")
print("categories replaced")