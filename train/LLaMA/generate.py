'''
# TODO | Generation script after training.
'''

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from transformers import PreTrainedTokenizerFast
 



