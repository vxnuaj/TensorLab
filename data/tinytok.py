'''

Utility Functions to preprocess the TinyStories dataset.

'''

import os
import requests
from typing import List, Union, Tuple

import torch
import pandas as pd
import pyarrow as pq
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


def stream_parquet_texts(file_paths: List[str]):
    """
    Stream text strings from multiple Parquet files in batches.

    This generator function reads the "text" column in each supplied Parquet file
    in batches of 10,000 rows and yields individual text strings. It is suitable
    for large datasets that cannot be loaded entirely into memory.

    Parameters:
        file_paths (List[str]): A list of paths to Parquet files containing a "text" column.

    Yields:
        str: Individual text strings from the "text" column.
    """
    for path in file_paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=10000):
            df = batch.to_pandas()
            yield from df['text'].tolist()


def data_process(files: List[str],
                 eos_str: Union[str, None] = None,
                 return_single_str: bool = False,
                 return_list_str: bool = False,
                 processes: int = 0) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
    """
    Process text data from Parquet files with options for formatting and parallel loading.

    Reads the list of Parquet files into a DataFrame (either sequentially or with multiple processes),
    optionally appending an end-of-sequence (EOS) string. It can return the data as a DataFrame,
    a tuple with the DataFrame and a single concatenated string, or just a list of strings.

    Parameters:
        files (List[str]): List of file paths to Parquet files.
        eos_str (str, optional): A string to append to each sequence in the "text" column.
        return_single_str (bool, optional): If True, return a tuple (DataFrame, concatenated string).
        return_list_str (bool, optional): If True, return the text column as a list.
        processes (int, optional): Number of processes to use for parallel reading. If 0, reads sequentially.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, str], List[str]]:
            - If return_list_str is True, returns a list of text strings.
            - If return_single_str is True, returns a tuple (DataFrame, concatenated string).
            - Otherwise, returns a DataFrame with the processed text data.
    """
    tqdm.pandas()
    if processes > 0:
        with Pool(processes=processes) as pool:
            dfs = list(tqdm(pool.map(pd.read_parquet, files),
                            total=len(files),
                            desc="Reading Files"))
    else:
        dfs = [pd.read_parquet(f) for f in tqdm(files, desc="Reading Files")]

    data = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    if eos_str:
        print("Appending EOS string to each entry.")
        data['text'] = data['text'] + eos_str

    if return_list_str:
        print(f"Returning list of {len(data)} text entries.")
        return data['text'].tolist()
    if return_single_str:
        print(f"Concatenating {len(data)} sequences into a single string.")
        return data, "".join(data['text'])
    return data


def train_new_tokenizer_bpe(data: List[str], 
                            vocab_size: int, 
                            special_tokens: List[str], 
                            save_path: Union[str, None] = None) -> Tokenizer:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on a list of text strings.

    Initializes a BPE tokenizer with ByteLevel pre-tokenization and decoding, trains it on the provided
    text data with a specified vocabulary size and special tokens, and saves the tokenizer if a path is given.

    Parameters:
        data (List[str]): List of text strings for training the tokenizer.
        vocab_size (int): Desired vocabulary size.
        special_tokens (List[str]): List of special tokens to include.
        save_path (str, optional): File path to save the trained tokenizer.

    Returns:
        Tokenizer: The trained BPE tokenizer.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, show_progress=True)
    
    tokenizer.train_from_iterator(data, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer


def tokenize_chunk(chunk_data: List[str], tokenizer: Tokenizer) -> List[List[int]]:
    """
    Tokenize a chunk of text data using the provided tokenizer.

    This helper function processes a list of strings by encoding them into a list of token IDs.

    Parameters:
        chunk_data (List[str]): List of text strings to be tokenized.
        tokenizer (Tokenizer): The tokenizer to encode the strings.

    Returns:
        List[List[int]]: A list where each element is a list of token IDs representing a text string.
    """
    encodings = tokenizer.encode_batch(chunk_data)
    token_ids = [encoding.ids for encoding in encodings]
    return token_ids


def tokenize(data: pd.DataFrame, 
             tokenizer: Tokenizer, 
             flat_tensor: bool = True, 
             processes: int = 1) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Tokenize text data from a DataFrame, supporting parallel processing.

    The function splits the "text" column into chunks, tokenizes each chunk in parallel,
    and then flattens the result into a single tensor or returns a list of tensors per sequence.

    Parameters:
        data (pd.DataFrame): DataFrame with a "text" column containing text strings.
        tokenizer (Tokenizer): Tokenizer for encoding text into token IDs.
        flat_tensor (bool, optional): If True, returns a single 1D tensor of all token IDs.
                                      If False, returns a list of 1D tensors for each text.
        processes (int, optional): Number of processes to use for parallel tokenization.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]:
            - 1D tensor of token IDs if flat_tensor is True.
            - List of 1D tensors, one per text entry, if flat_tensor is False.
    """
    print(f"Tokenizing {len(data)} strings using {processes} process(es).")
    chunk_size = max(1, len(data) // processes)
    chunks = [data['text'].iloc[i:i + chunk_size].tolist() for i in range(0, len(data), chunk_size)]
    
    with Pool(processes=processes) as pool:
        tokenize_partial = partial(tokenize_chunk, tokenizer=tokenizer)
        results = list(tqdm(pool.map(tokenize_partial, chunks),
                            total=len(chunks),
                            desc="Tokenizing in parallel"))
    
    token_ids = [token for chunk in results for token in chunk]
    
    if flat_tensor:
        total_tokens = sum(len(ids) for ids in token_ids)
        data_flat = torch.zeros(total_tokens, dtype=torch.long)
        offset = 0
        for ids in tqdm(token_ids, desc="Creating flat tensor"):
            data_flat[offset:offset + len(ids)] = torch.tensor(ids, dtype=torch.long)
            offset += len(ids)
        return data_flat
    
    return [torch.tensor(ids, dtype=torch.long) for ids in token_ids]


def generate_sequence_batch(data: torch.Tensor, 
                            start_indices: List[int], 
                            context_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of input-target sequences from tokenized data.

    For each start index, extracts a sequence of length `context_len` from the tokenized data
    and shifts it by one position for the target sequence.

    Parameters:
        data (torch.Tensor): 1D tensor containing token IDs.
        start_indices (List[int]): List of starting indices for sequences.
        context_len (int): Length of each sequence.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple (X, y) where X contains sequences and y contains
                                             the corresponding target sequences.
    """
    X_list, y_list = [], []
    for start_idx in start_indices:
        if start_idx + context_len + 1 > len(data):
            break
        X_list.append(data[start_idx:start_idx + context_len])
        y_list.append(data[start_idx + 1:start_idx + context_len + 1])
    return torch.stack(X_list, dim=0), torch.stack(y_list, dim=0)


def create_train_sequences_gen(data: torch.Tensor, 
                               context_len: int, 
                               seq_tensor_size: Union[int, None] = None, 
                               max_toks: Union[int, None] = None, 
                               processes: int = 4):
    """
    Generate training sequence batches from tokenized data.

    Depending on the input, the function either:
      - Yields batches of (X, y) tensors using multiple threads if seq_tensor_size is provided.
      - Returns a single tuple containing all sequences otherwise.

    A step size is derived from max_toks and the context length to slide over the token tensor.

    Parameters:
        data (torch.Tensor): 1D tensor containing token IDs.
        context_len (int): Length of each training sequence.
        seq_tensor_size (int, optional): Number of sequences per batch. If provided, data is yielded in batches.
        max_toks (int): Maximum number of tokens to process. This value is required.
        processes (int, optional): Number of threads for parallel batch generation.

    Yields or Returns:
        - If seq_tensor_size is specified: Yields tuples (X, y) for each batch.
        - Otherwise: Returns a tuple (X, y) containing all sequences.
    """
    assert max_toks is not None, 'Parameter max_toks must be provided'
    max_toks = min(max_toks, len(data))
    num_sequences = max_toks // context_len
    step_size = (len(data) - context_len) // num_sequences
    print(f"Generating {num_sequences} sequences with a step size of {step_size}.")

    if isinstance(seq_tensor_size, int):
        batch_size = seq_tensor_size
        start_indices = [i * step_size for i in range(num_sequences)]
        batches = [start_indices[i:i + batch_size] for i in range(0, len(start_indices), batch_size)]
        with ThreadPoolExecutor(max_workers=processes) as executor:
            generate_partial = partial(generate_sequence_batch, data, context_len=context_len)
            for batch_result in tqdm(executor.map(generate_partial, batches),
                                     total=len(batches),
                                     desc="Generating sequence batches"):
                X_batch, y_batch = batch_result
                yield X_batch, y_batch
    else:
        X_all, y_all = [], []
        for i in tqdm(range(num_sequences), desc=f"Creating {num_sequences} sequences"):
            start_idx = i * step_size
            if start_idx + context_len + 1 > len(data):
                break
            X_all.append(data[start_idx:start_idx + context_len])
            y_all.append(data[start_idx + 1:start_idx + context_len + 1])
        return torch.stack(X_all, dim=0), torch.stack(y_all, dim=0)


def create_val_sequences(data: List[torch.Tensor],
                         batch_first: bool = True,
                         padding_value: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create padded validation sequences from tokenized data.

    From a list of variable-length tensors, this function builds the input (X) and target (y)
    sequences by shifting the tokens by one. It then pads all sequences to a uniform length.

    Parameters:
        data (List[torch.Tensor]): List of 1D tensors representing tokenized sequences.
        batch_first (bool, optional): If True, resulting tensors have shape (batch_size, max_seq_length).
        padding_value (int, optional): The value used for padding shorter sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple (X_padded, y_padded) of padded input and target tensors.
    """
    X = [seq[:-1] for seq in data]
    y = [seq[1:] for seq in data]
    X_padded = pad_sequence(sequences=X, batch_first=batch_first, padding_value=float(padding_value))
    y_padded = pad_sequence(sequences=y, batch_first=batch_first, padding_value=float(padding_value))
    return X_padded, y_padded

def download_tinystories(save_dir: str = 'data/data'):
    """
    Download multiple Parquet files and save them to a specified directory.

    The function creates the target directory if it does not exist, downloads each file from
    its URL, writes it to a temporary file, and renames it to a new designated filename.

    Parameters:
        save_dir (str, optional): Directory where the downloaded files will be saved.
                                  Defaults to 'data/data'.
    """
    os.makedirs(save_dir, exist_ok=True)

    urls = [
        ('train1.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet?download=true'),
        ('train2.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00001-of-00004-5852b56a2bd28fd9.parquet?download=true'),
        ('train3.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00002-of-00004-a26307300439e943.parquet?download=true'),
        ('train4.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00003-of-00004-d243063613e5a057.parquet?download=true'),
        ('validation.parquet', 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet?download=true'),
    ]

    for new_name, url in tqdm(urls, desc="Downloading files"):
        tmp_name = url.split("/")[-1].split("?")[0]
        tmp_path = os.path.join(save_dir, tmp_name)
        final_path = os.path.join(save_dir, new_name)

        response = requests.get(url)
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        os.rename(tmp_path, final_path)
