import yaml 
import argparse

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def add_dict_to_namespace(namespace, args_dict):
    for key, value in args_dict.items():
        setattr(namespace, key, value)

def arguments_from_file(config_file_path: str) -> argparse.Namespace:
    # read config file
    args_dict = load_yaml(config_file_path)

    # create argparse Namspace object
    args = argparse.Namespace()

    # add config dictionary into argparse namespace
    add_dict_to_namespace(args, args_dict)

    return args

if __name__ == "__main__":
    file_path= "./library_modules/osmosis_sample.yaml"
    args = arguments_from_file(file_path)
    args.image_size = args.unet_model['image_size']
    # print(args)
    # print(args.image_size)
