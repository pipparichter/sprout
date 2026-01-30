import pandas as pd 
import argparse
from src.embed import Embedder 
import numpy as np 
from src.model import MLP, Tree
from src.dataset import Dataset, split
import os 
# import pickle
import json

# sbatch --mail-user prichter@berkeley.edu --mail-type ALL --mem 100GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed --input-path ./data/dataset_train.csv"
# sbatch --mail-user prichter@berkeley.edu --mail-type ALL --mem 100GB --partition gpu --gres gpu:1 --time 24:00:00 --wrap "embed --input-path ./data/dataset_test.csv"
def embed():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', default=None, type=str)
    args = parser.parse_args()

    output_path = args.output_path if (args.output_path is not None) else args.input_path.replace('.csv', '.h5')

    # Sort the sequences in order of length, so that the longest sequences are first. This is so that the 
    # embedder works more efficiently, and if it fails, it fails early in the embedding process.
    df = pd.read_csv(args.input_path, index_col=0)
    df = df.iloc[np.argsort(df.seq.apply(len))[::-1]]

    store = pd.HDFStore(output_path, mode='w', table=True) # Should confirm that the file already exists. 
    store.put('metadata', df, format='table', data_columns=None)

    inputs = list(zip(df.index.values.tolist(), df.seq.values.tolist()))
    embedder = Embedder()
    embeddings = embedder(inputs)
    store.put('embeddings', pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 
    store.close()
    print(f'embed: Embeddings saved to {output_path}.')



MLP_PARAMS = {'epochs':20, 'alpha':0.5, 'lr':1e-4, 'batch_size':64}
TREE_PARAMS = {}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--model-class', default='mlp')
    parser.add_argument('--model-id', default='0')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    model = MLP(model_id=args.model_id)
    dataset = Dataset.from_hdf(args.dataset_path)
    dataset_train, dataset_test = split(dataset)

    info = dict()
    info['results'] = model.fit((dataset_train, dataset_test), **MLP_PARAMS)
    info.update(MLP_PARAMS)

    model_path = os.path.join(args.output_dir, f'{model.model_id}.pkl')
    summary_path = os.path.join(args.output_dir, f'{model.model_id}.summary.json')
    model.save(path=model_path)

    with open(summary_path, 'w') as f:
        json.dump(info, f)


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    model = MLP.load(args.model_path)
    dataset = Dataset.from_hdf(args.dataset_path)

    results = model.predict(dataset)
    results_path = os.path.join(args.output_dir, f'{model.model_id}.results.json')
    for metric in ['prc_auc_0', 'prc_auc_1', 'roc_auc_0', 'roc_auc_1']:
        print(metric, results[metric])

    with open(results_path, 'w') as f:
        json.dump(results, f)

