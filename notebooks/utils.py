import pandas as pd 
import numpy as np 
import re
import matplotlib.pyplot as plt 
import seaborn as sns 

import matplotlib.pyplot as plt 
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import chisquare, mannwhitneyu
from scipy.stats.contingency import expected_freq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os 
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import json
import requests 


WIDTH_MM = 85 # Half-page figure width for BMC Bioinformatics. 
WIDTH_IN = WIDTH_MM / 25.4
MAX_HEIGHT_MM = 225 # Maximum figure height for BMC Bioinformatics. 
MAX_HEIGHT_IN = MAX_HEIGHT_MM / 25.4

DPI = 300
    
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 3


def get_antifam_ids(path:str='../data/antifam_ids.json'):
    '''Get metadata for all AntiFam families from the InterPro website.'''
    antifam_ids = []
    result = json.loads(requests.get('https://www.ebi.ac.uk/interpro/api/entry/antifam/').text)
    pbar = tqdm(total=result['count'], desc='AntiFam.get_antifams')
    while True: # Only returns 20 hits at a time, so need to paginate using the 'next' field. 
        antifam_ids += [{'id':entry['metadata']['accession'], 'description':entry['metadata']['name']} for entry in result['results']]
        pbar.update(len(result['results']))
        if result['next'] is None:
            break
        result = json.loads(requests.get(result['next']).text)
    with open(path, 'w') as f:
        json.dump(antifam_ids, f)
    print(f'AntiFam.get_antifams: IDs for {len(antifam_ids)} AntiFam families written to {path}')