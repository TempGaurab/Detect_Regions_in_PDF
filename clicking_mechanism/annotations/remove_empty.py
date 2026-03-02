import os
import json
import glob

ANNOTATIONS_DIR = os.path.dirname(os.path.abspath(__file__))

deleted = 0
for json_path in glob.glob(os.path.join(ANNOTATIONS_DIR, '*.json')):
    with open(json_path) as f:
        data = json.load(f)
    if not data.get('checkboxes') and not data.get('lines') and not data.get('boxes'):
        os.remove(json_path)
        print(f'Deleted: {json_path}')
        deleted += 1

print(f'\nDone. {deleted} empty file(s) deleted.')