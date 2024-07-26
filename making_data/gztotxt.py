import gzip

def extract_gz_file(gz_file_path, txt_file_path):
    with gzip.open(gz_file_path, 'rt') as gz_file:
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(gz_file.read())

# Usage example
gz_file_path = 'lan-mp-instances/indtrans40-100-1000e.gz'
txt_file_path = 'parameters_txt/parameters40-100-1000e.txt'
extract_gz_file(gz_file_path, txt_file_path)