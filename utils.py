import ConfigParser
import pickle as pkl


def load_config_from_file(filename, main_key, config=None):
    cf = ConfigParser.ConfigParser()    
    cf.read(filename)
    if config is None:
        config = {}
    config_int_keys = ["max_sentence_length1", "max_sentence_length2", "emb_dim",
        "voc_size","nb_classes", "lstm_dim", "nb_epoches", "decay_start"]
    config_float_keys = ["lr", "lr_decay"]
    config_string_keys = ["preprocess", "ave_mode"]
    for int_key in config_int_keys:
        if cf.has_option(main_key, int_key):
            config[int_key] = cf.getint(main_key, int_key)
    for float_key in config_float_keys:
        if cf.has_option(main_key, float_key):
            config[float_key] = cf.getfloat(main_key, float_key)
    for str_key in config_string_keys:
        if cf.has_option(main_key, str_key):
            config[str_key] = cf.get(main_key, str_key)

    return config

def load_snli_data():
    data_path = "../data/processed_snli.pkl"
    f = open(data_path, 'rb')
    data = pkl.load(f)
    f.close()

    train = data["train"]
    dev = data["dev"]
    test = data["test"]
    word2idx = data["voc"]

    return train, dev, test, word2idx



if __name__ == "__main__":
    config = load_config_from_file("config", "snli")
    config = load_config_from_file("config", "train_snli", config)
    print(config)