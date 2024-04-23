
import sys
from os import path
import pandas as pd
import json
from ordered_set import OrderedSet
from PIL import Image



def extractData(file_path: str,images_path:str=None):
    
    images = {}
    # Open file
    with open(file_path, 'r') as file:
        data = json.load(file)
    possible_labels = None
    rows = []
    # Check using first entry wether data has labels and pair images

    if('labels' in data[0].keys()):
        label_key = 'labels'
        possible_labels = OrderedSet(label for entry in data for label in entry['labels'])
    elif('label' in data[0].keys()):
        label_key= 'label'
        possible_labels = OrderedSet(entry['label'] for entry in data)
        main_label = possible_labels[1]
        
    else:
        label_key=None

    if('image' not in data[0].keys()):
        use_images=False
    else:
        use_images=True
        if(not images_path):
            raise Exception("Error loaded file points to images, but you have not provided a path to the folder containing them : images_path is None")


    # Iterate through each entry in the Json
    for entry in data:
        
        text = entry['text']
        # Create a new row for the final dataframe
        row = {}
        row['text'] = text
        current_id = entry['id']
        row['id'] = current_id
        
        
        if(label_key):
            # If labels are binary treat accordingly
            if(label_key == 'label'):
                if(main_label == entry[label_key]):
                    row[main_label] = 1
                else:
                    row[main_label] = 0

            else:
                labels = entry[label_key]
                for label in possible_labels:
                    if(label in labels):
                        row[label] = 1
                    else:
                        row[label] = 0

        rows.append(row)
        if(use_images):
            path_to_image = path.join(images_path,entry['image'])
            images[current_id] = Image.open(path_to_image)
            

    df = pd.DataFrame(rows)


    return df,images,possible_labels



def augment_text_random_insertion():
    return


def augment_text_synonym_replacement():
    return

def augment_text_random_swap():
    return

def augment_text_random_deletion():
    return

def augment_text_back_translation():
    return
