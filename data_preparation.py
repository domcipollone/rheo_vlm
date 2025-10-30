import torch 
import json 
from pathlib import Path 
import math 
from PIL import Image 
from datasets import Dataset, DatasetDict


if torch.cuda.is_available() and (torch.cuda.memory_allocated() // 1024 // 1024) > 10: 
    print(f"{torch.cuda.memory_allocated() // 1024 // 1024} MB currently allocated")
    print(f"{torch.cuda.memory_reserved() // 1024 // 1024} MB currently reserved")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()