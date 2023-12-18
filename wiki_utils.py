import os
import hashlib
import json

from tensorflow.keras.utils import get_file
from tqdm.notebook import tqdm
import mwparserfromhell
import xml.sax

import data_processor


def md5(fname, compressed_path):
    """Returns md5 hash of file with name fname"""
    hash_md5 = hashlib.md5()
    with open(compressed_path + fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_md5(checksums, files_to_download, compressed_path):
    with open(checksums) as file:
        data = file.read()

    md5_lst = data.split("\n")
    md5_lst = [x.split("  ") for x in md5_lst][:-1]
    md5_lst = [file for file in md5_lst if ".xml-p" in file[1] and not "multistream" in file[1] and not "meta" in file[1]]

    downloaded_md5 = []
    if [x[1] for x in md5_lst] == files_to_download:
        
        for file in tqdm(files_to_download):
            downloaded_md5.append(md5(file, compressed_path))
            
        if [x[0] for x in md5_lst] == downloaded_md5:
            print("Downloads verified by MD5 checksums")
        else:
            print("Download was faulty, the following files could not be verified:")
            
            for file in [x for x in downloaded_md5 if x not in md5_lst]:
                print(file)
    else:
        print("Files to downloaded are not equal to md5 checksum files")


def download_wikipedia(data_folder, files_to_download, dump_url):
    data_paths = []
    file_info = []

    # create project dirs
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    wiki_path = data_folder + "wikipedia\\"
    if not os.path.exists(wiki_path):
        os.mkdir(wiki_path)
    compressed_path = wiki_path + "compressed\\"
    if not os.path.exists(compressed_path):
        os.mkdir(compressed_path)

    # Iterate through each file
    for file in files_to_download:
        path = compressed_path + file
        
        # Check to see if the path exists (if the file is already downloaded)
        if not os.path.exists(path):
            # If not, download the file
            data_paths.append(get_file(path, dump_url + file))
            # Find the file size in MB
            file_size = os.stat(path).st_size / 1e6
            
            # Find the number of articles
            file_articles = int(file.split('p')[-1].split('.')[-2]) - int(file.split('p')[-2])
            file_info.append((file, file_size, file_articles))
            
        # If the file is already downloaded find some information
        else:
            data_paths.append(path)
            # Find the file size in MB
            file_size = os.stat(path).st_size / 1e6
            
            print(f"Found File {file}, size: {round(file_size, 2)} MB")
            
            # Find the number of articles
            file_number = int(file.split('p')[-1].split('.')[-2]) - int(file.split('p')[-2])
            file_info.append((file.split('-')[-1], file_size, file_number))

    return data_paths, file_info


def read_json(file_path):
    """Read in json data from `file_path`"""
    
    data = []
    # Open the file and load in json
    with open(file_path, 'r') as fin:
        for l in fin.readlines():
            data.append(json.loads(l))
            
    return data

