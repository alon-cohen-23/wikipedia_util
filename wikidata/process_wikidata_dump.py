# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:31:53 2023

@author: DEKELCO
"""
import os
import indexed_bzip2 as ibz2
import pickle
from pathlib import Path

def open_bzip_cache_offsets(filepath):
    p_path = Path(filepath)
    p_offsets_path = p_path.with_suffix('').with_name(p_path.stem + '_block_offsets.dat')

    # Check if the block offsets file already exists
    if p_offsets_path.exists():
        # If the cached file exists, load the block offsets and set them for the opened .bz2 file
        with open(p_offsets_path, 'rb') as offsets_file:
            loaded_block_offsets = pickle.load(offsets_file)
        file = ibz2.open(filepath, parallelization=os.cpu_count())
        file.set_block_offsets(loaded_block_offsets)
    else:
        # If the cached file doesn't exist, calculate block offsets and create the cached file
        file = ibz2.open(filepath, parallelization=os.cpu_count())
        block_offsets = file.block_offsets()
        with open(p_offsets_path, 'wb') as offsets_file:
            pickle.dump(block_offsets, offsets_file)

    return file

if __name__ == "__main__":
    filepath = 'latest-all.json.bz2'
    wikidata_file = open_bzip_cache_offsets(filepath)
    # Now you can use the 'opened_file' object for further operations
    # ...
