import pandas as pd 
import arparse
from src.embed import ESMEmbedder

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

    embedder = ESMEmbedder()
    embeddings = embedder(df.seq.values.tolist())
    store.put('embeddings', pd.DataFrame(embeddings, index=df.index), format='table', data_columns=None) 
    store.close()
    print(f'embed: Embeddings saved to {output_path}.')
