from tqdm import tqdm
import numpy as np
import torch 
import esm
from esm import BatchConverter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Embedder():

    def __init__(self):

        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.to(DEVICE) # Move model to GPU.
        self.model.half() # Convert weights to float16. 
        self.model.eval() # Set model to evaluation model.

        self.batch_converter = self.alphabet.get_batch_converter()

    def embed_batch(self, batch) -> torch.FloatTensor:

        # BatchConverter expects a list of tuples of the form: [(id, seq), (id, seq)]
        batch_labels, batch_seqs, batch_tokens = self.batch_converter(batch)
        batch_lengths = [len(seq) for (_, seq) in batch]
        batch_tokens = batch_tokens.to(DEVICE)

        # Autocast automatically determines which operations are safe to use float16 in, and which should keep float32 for numerical stability.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = self.model(batch_tokens, repr_layers=[33], return_contacts=False) # Layer 33 is the last layer of the model. 
            # This has shape tensor(batch_size, length, dim). 
            outputs = outputs['representations'][33]
            outputs = [outputs[i, :length, :].mean(dim=0) for i, length in enumerate(batch_lengths)] # Remove the padding and mean-pool.
        return outputs
   
    def __call__(self, inputs, max_aa_per_batch:int=10000):

        inputs = [(id_, seq.replace('*', '')) for (id_, seq) in inputs] # Just make sure there's no terminal asterisk.  

        embeddings = list()
        aa_count = 0
        batch = list()
        for (id_, seq) in tqdm(inputs, desc='PLMEmbedder.__call__'):

            batch.append((id_, seq))
            aa_count += len(seq)

            if aa_count > max_aa_per_batch:
                embeddings += self.embed_batch(batch)
                batch, aa_count = list(), 0

        # Handles the case in which the minimum batch size is not reached.
        if aa_count > 0:
            embeddings += self.embed_batch(batch)

        embeddings = torch.cat([torch.unsqueeze(embedding, 0) for embedding in embeddings]).cpu().float()
        return embeddings.numpy()

