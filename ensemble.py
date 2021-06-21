import os
import numpy as np
import pandas as pd

csv_dir = '/opt/ml/code/output/ensemble'
csv_list = os.listdir(csv_dir)
print(csv_list)

preds = np.zeros(744)

for csv in csv_list:
    csv_path = os.path.join(csv_dir, csv)
    df = pd.read_csv(csv_path)
    preds += df['prediction'].values
preds /= len(csv_list)

write_path = '/opt/ml/code/output/ensemble_output/ensemble.csv'
if not os.path.exists('/opt/ml/code/output/ensemble_output'):
    os.makedirs('/opt/ml/code/output/ensemble_output')    
    
with open(write_path, 'w', encoding='utf8') as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(preds):
        w.write('{},{}\n'.format(id,p))
