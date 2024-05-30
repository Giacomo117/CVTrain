import os
import requests
import tarfile
import patoolib
from tqdm import tqdm

def download_and_extract_dataset(url, file_path, extract_path):
    if os.path.exists(extract_path + "/YawDD dataset"):
        return;
    # Check if the file already exists
    print("Checking if the dataset already exists")
    if not os.path.exists(file_path):
        print("Downloading the dataset")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    else:
        print("Dataset already exists")

    extracted_file_path = extract_path + '/'+ file_path.split('.')[0] + "/" + "YawDD dataset.rar"
    if not os.path.exists(extracted_file_path):
        print("Download completed, extracing the dataset")
        with tarfile.open(file_path) as tar:
            members = tar.getmembers()
            progress_bar = tqdm(total=len(members), unit='files')
            for member in members:
                tar.extract(member, path=extract_path)
                progress_bar.update()
            progress_bar.close()
    else:
        print("Dataset already extracted")
    
    if not os.path.exists(extract_path + "/YawDD dataset"):
        print("Extracting the rar file")
        patoolib.extract_archive(extracted_file_path, outdir=extract_path)

    

if __name__ == '__main__':
    # Usage
    url = 'http://skuld.cs.umass.edu/traces/mmsys/2014/user06.tar'
    file_path = 'user06.tar'
    extract_path = './yawdd'
    download_and_extract_dataset(url, file_path, extract_path)