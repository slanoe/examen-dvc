import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(object_url)
            if response.status_code == 200:
                content = response.text
                text_file = open(output_file, "wb")
                text_file.write(content.encode('utf-8'))
                text_file.close()
            else:
                print(f'Error accessing the object {input_file}:', response.status_code)
                
def main(raw_data_relative_path="./data/raw", 
        filenames = ["raw.csv"],
        bucket_folder_url= "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"          
        ):
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()