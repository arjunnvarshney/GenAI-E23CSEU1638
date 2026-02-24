import os
import tarfile
import urllib.request

def download_facades():
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    output_path = "facades.tar.gz"
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(os.path.join(data_dir, "facades")):
        print("Downloading Facades dataset...")
        urllib.request.urlretrieve(url, output_path)
        print("Extracting dataset...")
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        os.remove(output_path)
        print("Done.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_facades()
