from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM
from transformers import T5EncoderModel, T5Tokenizer 
from tqdm import tqdm
import numpy as np
import torch 


class PLMEmbedder():

    def __init__(self, model=None, tokenizer=None, checkpoint:str=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.from_pretrained(checkpoint)
        self.model.to(self.device) # Move model to GPU.
        self.model.eval() # Set model to evaluation model.
        self.tokenizer = tokenizer.from_pretrained(checkpoint, do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)

    def embed_batch(self, seqs:list) -> torch.FloatTensor:

        # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
        # The tokenizer defaults mean that add_special_tokens=True and padding=True is equivalent to padding='longest'
        inputs = {k:torch.tensor(v).to(self.device) for k, v in self.tokenizer(seqs, padding=True).items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
   
    def __call__(self, seqs:list, max_aa_per_batch:int=1000):

        seqs = self._preprocess(seqs)

        embs = list()
        aa_count = 0
        batch_seqs = list()
        for seq in tqdm(seqs, desc='PLMEmbedder.__call__'):

            batch_seqs.append(seq)
            aa_count += len(seq)

            if aa_count > max_aa_per_batch:
                outputs = self.embed_batch(batch_seqs)
                embs += self._postprocess(outputs, seqs=batch_seqs)
                batch_seqs = list()
                aa_count = 0

        # Handles the case in which the minimum batch size is not reached.
        if aa_count > 0:
            outputs = self.embed_batch(batch_seqs)
            embs += self._postprocess(outputs, seqs=batch_seqs)

        embs = torch.cat([torch.unsqueeze(emb, 0) for emb in embs]).float()
        return embs.numpy()



class ESMEmbedder(PLMEmbedder):
    checkpoints ={'3b':'facebook/esm2_t36_3B_UR50D', '650m':'facebook/esm2_t33_650M_UR50D'}

    @staticmethod
    def _mean_pool(emb:torch.FloatTensor, seq:str) -> torch.FloatTensor:
        emb = emb[1:len(seq) + 1] # First remove the CLS token from the mean-pool, as well as any padding... 
        emb = emb.mean(dim=0)
        return emb 

    def __init__(self, model_size:str='650m'):

        models = {'gap':EsmModel, 'log':EsmForMaskedLM, 'cls':EsmModel}
        checkpoint = ESMEmbedder.checkpoints.get(model_size)

        super(ESMEmbedder, self).__init__(model=models[pooler], tokenizer=AutoTokenizer, checkpoint=checkpoint)
        self.pooler = poolers.get(pooler, None)

    def _preprocess(self, seqs:list):
        # Based on the example Jupyter notebook, it seems as though sequences require no real pre-processing for the ESM model.
        return [seq.replace(r'*', '') for seq in seqs] # Just make sure there's no terminal asterisk.  

    def _postprocess(self, outputs:torch.FloatTensor, seqs:list=None):
        '''Mean-pool the output embedding'''
        # Transferring the outputs to CPU and reassigning should free up space on the GPU. 
        # https://discuss.pytorch.org/t/is-the-cuda-operation-performed-in-place/84961/6 
        outputs = outputs.last_hidden_state.cpu() # if (self.model_name == 'pt5') else outputs.pooler_output
        outputs = [self._mean_pool(emb, seq) for emb, seq in zip(outputs, seqs)]
        return outputs       
