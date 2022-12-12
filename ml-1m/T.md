from torch.utils.data import Dataset, DataLoader
from parameters import *
from transformer import *

import torch
import sentencepiece as spm
import numpy as np
import pandas as pd
import heapq
import warnings
from pathlib import Path

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem



def build_model(model_type):
    print(f"{model_type}molecular model is building...")
    #print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{SP_DIR}/{model_type}_src_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{SP_DIR}/{model_type}_trg_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word

    #print(f"The size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")

    return Transformer(src_vocab_size=len(src_i2w), trg_vocab_size=len(trg_i2w)).to(device)


def make_mask(src_input, trg_input):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask

# custom data structure for beam search method
class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")

class PriorityQueue():

    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)


#################
# Preprocessing of input SMILES

def getSmarts(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        symbol = '['+atom.GetSmarts()
        if nHs:
            symbol += 'H'
            if nHs>1:
                symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        symbol += ';D%d'%deg
        symbol += "]"
        symbols.append(symbol)
    try:
        smart = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    except (ValueError, RuntimeError) as ve:
        print('atom to use error or precondition bond error')
        return
    return smart


def getAtomEnvs(smiles, radii=[0, 1], radius=1, nbits=1024, rdLogger=False):
    """
    A function to extract atom environments from the molecular SMILES.
    Parameters
    ----------
    smiles: str
        Molecular SMILES
    radii: list
        list of radii you would like to obtain atom envs.
    radius: int
        radius of MorganFingerprint
    nbits: int
        size of bit vector for MorganFingerprint
    Returns
    -------
    tuple
        a list of atom envs and a string type of this list
    """

    assert max(radii) <= radius, f"the maximum of radii should be equal or lower than radius, but got {max(radius)}"

    RDLogger.EnableLog('rdApp.*') if rdLogger else RDLogger.DisableLog('rdApp.*')
    molP = Chem.MolFromSmiles(smiles.strip())
    if molP is None:
        if rdLogger:
            warnings.warn(f"There is a semantic error in {smiles}")
        #raise Exception (f"There is a semantic error in {smiles}")
        return None

    sanitFail = Chem.SanitizeMol(molP, catchErrors=True)
    if sanitFail:
        if rdLogger:
            warnings.warn(f"Couldn't sanitize: {smiles}")
        #raise Exception (f"Couldn't sanitize: {smiles}")
        return None

    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molP,radius=radius, nBits=nbits, bitInfo=info)# condition can change

    info_temp = []
    for bitId,atoms in info.items():
        exampleAtom,exampleRadius = atoms[0]
        description = getSmarts(molP,exampleAtom,exampleRadius)
        info_temp.append((bitId, exampleRadius, description))

    #collect the desired output in another list
    updateInfoTemp = []
    for k,j in enumerate(info_temp):
        if j[1] in radii:                           # condition can change
            updateInfoTemp.append(j)
        else:
            continue

    tokens_str = ''
    tokens_list = []
    for k,j in enumerate(updateInfoTemp):
        tokens_str += str(updateInfoTemp[k][2]) + ' ' #[2]-> selecting SMARTS description
        tokens_list.append(str(updateInfoTemp[k][2]))  # condition can change

    #return tokens_list, tokens_str.strip()
    return tokens_str.strip()



#################
# Data loaders

def get_data_loader(model_type, file_name):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{model_type}_src_sp.model")
    trg_sp.Load(f"{SP_DIR}/{model_type}_trg_sp.model")

    print(f"Getting source/target {file_name} for {model_type} molecular...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{model_type}_{file_name}", 'r', encoding="utf-8") as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{model_type}_{file_name}", 'r', encoding="utf-8") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list, src_sp) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list, trg_sp) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list, src_sp):
    tokenized_list = []
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list
