import json
import pandas as pd
import numpy as np
from flatten_json import flatten


def dend_json_to_df(json_file):
    with open(json_file, 'r') as f:
        s = f.read()
        s = s.replace('\t', '')
        s = s.replace('\n', '')
        s = s.replace(',}', '}')
        s = s.replace(',]', ']')
        dend = json.loads(s)

    flatten_dend = flatten(dend)
    label, members, height, color, index, midpoint = [], [], [], [], [], []
    org_label, parent, leaf, cex, xpos = [], [], [], [], []
    dend_keys = list(flatten_dend.keys())

    for i, _ in enumerate(dend_keys):
        if i < 1:
            index = i
        if index < len(dend_keys):
            entry = dend_keys[index]
            if 'leaf_attribute' in entry:
                ind_0 = [i for i, x in enumerate(entry) if x == '0']
                tag = entry[:ind_0[-1]+2]
                key = tag + '_row'
                label.append(flatten_dend[key])
                key = tag + 'members'
                members.append(flatten_dend[key])
                key = tag + 'height'
                height.append(flatten_dend[key])
                key = tag + 'nodePar.col'
                color.append(flatten_dend[key])
                midpoint.append('')
                key = tag + 'nodePar.cex'
                cex.append(flatten_dend[key])
                leaf.append(True)
                number_ind = label[-1].find('_')
                xpos.append(np.float16(label[-1][:number_ind]))
                ind_child = [i for i, _ in enumerate(entry[:-8])
                             if entry[i:i+8] == 'children']
                key_parent = entry[:ind_child[-2]+10] + '_node_attributes_0__row'
                if key_parent in flatten_dend:
                    parent.append(flatten_dend[key_parent])
                else:
                    parent.append('')
                index += 21
            if 'node_attribute' in entry:
                ind_0 = [i for i, x in enumerate(entry) if x == '0']
                tag = entry[:ind_0[-1]+2]
                key = tag + '_row'
                label.append(flatten_dend[key])
                key = tag + 'members'
                members.append(flatten_dend[key])
                key = tag + 'height'
                height.append(flatten_dend[key])
                color.append('')
                key = tag + 'midpoint'
                midpoint.append(flatten_dend[key])
                cex.append('')
                leaf.append(False)
                xpos.append(0.)
                ind_child = [i for i, _ in enumerate(entry[:-8])
                             if entry[i:i + 8] == 'children']
                if len(ind_child) > 0:
                    if len(ind_child) > 1:
                        key_parent = entry[:ind_child[-2] + 10] + \
                                     '_node_attributes_0__row'
                    else:
                        key_parent = 'node_attributes_0__row'
                    parent.append(flatten_dend[key_parent])
                else:
                    parent.append('')
                index += 15

    # find x position for all non leaf nodes
    x = np.array(xpos)
    for i, l in enumerate(label):
        if not leaf[i]:
            parent_ind = np.where(np.array(parent) == l)[0]
            x[i] = np.mean(x[parent_ind])

    # build a dataframe from the flatten dendrogram
    dend_df = pd.DataFrame({'x': list(x),
                            'y': height,
                            'cex': cex,
                            'col': color,
                            'members': members,
                            'midpoint': midpoint,
                            'height': height,
                            'leaf': leaf,
                            'label': label,
                            'parent': parent})

    # reverse the order nodes in the dataframe
    dend_df = dend_df.iloc[::-1].reset_index(drop=True)

    # replace empty values with nan
    dend_df = dend_df.replace(r'', np.NaN)

    # start the dataframe index from 1
    dend_df.index += 1

    return dend_df

