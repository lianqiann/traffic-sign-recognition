import torch
import torchvision.models as models
from tqdm import tqdm
from model import BaselineNet, TrafficSignNet, Stn
import pandas as pd
import pickle
from fooler import *
import argparse


def load_data(data_path):
    # Load pickled data
    # train_file = data_path+'train.pickle'
    # valid_file = data_path+'valid.pickle'
    # test_file = data_path+'test.pickle'

    # with open(train_file, mode='rb') as f:
    #     train = pickle.load(f)
    # with open(valid_file, mode='rb') as f:
    #     valid = pickle.load(f)
    # with open(test_file, mode='rb') as f:
    #     test = pickle.load(f)

    # X_train, y_train = train['features'], train['labels']
    # X_valid, y_valid = valid['features'], valid['labels']
    # X_test, y_test = test['features'], test['labels']

    # n_classes = len(set(y_train))

    sign_names = pd.read_csv(data_path+"signnames.csv")
    sign_names.set_index("ClassId")
    sign_classes = sign_names.SignName
    


    #return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), n_classes, sign_classes
    return sign_classes


def load_model(model_type, model_name):
    if model_type == 'baselinenet':
        model = BaselineNet().to(device)
    elif model_type == 'trafficsign':
        model = TrafficSignNet().to(device)

    model.load_state_dict(torch.load(args.PATH+f'models/{model_name}.pt', map_location = device))

    return model





if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description='Traffic sign recognition training script')
    parser.add_argument('--PATH', type=str, default='./',
                        help="Folder where data is located. train.p and vliad.p need to be found in the folder (default: data)")
    parser.add_argument('--model_name', type = str, default= '128baseline_model',
                        help="the name of the model: path/models/<model_name>.pt")
    parser.add_argument('--model_type', type = str, default= 'baselinenet',
                        help="the type of model, options: baselinenet, trafficsignnet")

    parser.add_argument('--n_epochs', type = int, default= 10,
                        help="the number of steps you want the bw works for")
    
    parser.add_argument('--image_name', type=str, default='class: 1.png', 
                        help='the name of the image: path/examples/<image_name>')
    parser.add_argument('--fooler_class', type=int, default=0,
                        help='the class we want the recognizer to be wrongly recognize to')
    parser.add_argument('--run_all', type=bool, default=True,
                        help='whether loop over all classes')

    args = parser.parse_args()



    

    data_path = args.PATH+'data/'
    sign_classes = load_data(data_path)

    device = 'cuda' if cuda_available else 'cpu'
    model = load_model(args.model_type, args.model_name)
    get_fooler_image(args, model,sign_classes, device)

    if args.run_all:
        for i in tqdm(range(1,43)):
            args.image_name = f'class: {i}.png'
            get_fooler_image(args, model,sign_classes, device)


    

    
    
  



