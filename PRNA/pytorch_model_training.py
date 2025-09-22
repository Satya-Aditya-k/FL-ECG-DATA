from train_12ECG_classifier import train_12ECG_classifier
from feats.features import *
import wfdb
from scipy.io import loadmat
import os
import numpy as np
import neurokit2 as nk
from model import CTN
from utils import *
from train_12ECG_classifier import train,validate, get_probs
from optimizer import NoamOpt
from tensorboardX import SummaryWriter
# stop warningd
import warnings
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
import tqdm
from scipy.signal import decimate,resample
warnings.filterwarnings("ignore")

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data
     

def import_key_data(path):
    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    ecg_data=[]
    header_data=[]
    for subdir2, dirs2, files2 in sorted(os.walk(path)):
        for filename in files2:
            if filename.endswith(".mat"):
                filepath = subdir2 + os.sep + filename
                data, header = load_challenge_data(filepath)
                ecg_data.append(data)
                header_data.append(header)
                labels.append(header[15][5:-1])
                ecg_filenames.append(filepath)
                gender.append(header[14][6:-1])
                age.append(header[13][6:-1])
    return gender, age, labels, ecg_filenames,ecg_data,header_data

def filter_labels(labels,dx_map_df):
    labels = [label.strip() for label in labels]
    #labels_first = [label.split(',')[0] if len(label.split(',')) > 1 else label for label in labels]
    labels_first = []
    label_indices = []
    for i,label in enumerate(labels):

        if len(label.split(',')) < 2:
            if label in dx_map_df['SNOMED CT Code'].values.astype(str):
                labels_first.append(label)
                label_indices.append(i)

        else:
            len_label = len(label.split(','))
            for j in range(len_label):
                if label.split(',')[j] in dx_map_df['SNOMED CT Code'].values.astype(str):
                    labels_first.append(label.split(',')[j])
                    label_indices.append(i)
                    #break
                # else:
                #     labels_first.append(label.split(',')[0])
                #     break
            
            # if label.split(',')[1] in dx_map_df['SNOMED CT Code'].values.astype(str):
            #     labels_first.append(label.split(',')[1])
            # else:
            #     labels_first.append(label.split(',')[0])

            
    label_count = {}
    for label in labels_first:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return labels_first, label_indices

def filter_codes(labels, dx_map_df):
    labels_filtered, label_indices = filter_labels(labels, dx_map_df)
    
    label_count_filtered = {}
    for label in labels_filtered:
        if label in label_count_filtered:
            label_count_filtered[label] += 1
        else:
            label_count_filtered[label] = 1
    print("no.of unique labelsin snomed:",len(label_count_filtered))
    print("len of filtered labels in snomed:",len(labels_filtered))
    return labels_filtered, label_indices

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal

def write_log(loc, fold, epoch, trn_loss, trn_auroc, val_loss, val_auroc):
    with open(loc/f'log_fold_{fold}.csv', 'a') as f:
        f.write(f'{epoch}, {trn_loss}, {trn_auroc}, {val_loss}, {val_auroc}\n')                    

def extract_templates(signal, rpeaks, before=0.2, after=0.4, fs=500):
    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(signal)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b > length:
            break

        # Append template list
        templates.append(signal[a:b])

        # Append new rpeaks list
        rpeaks_new = np.append(rpeaks_new, rpeak)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new

def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

# convert above code to a function
def create_ecg_segs(ecg_data_dict, labels_filtered, window, nb_windows, filter_bandwidth, polarity_check,fs_list,features_df):
    

    ecg_segs = []
    for i in features_df.index:
        id = features_df.loc[i,"id"]
        data = ecg_data_dict[id]
        
        if fs_list[i] > fs:
            data = decimate(data, int(fs_list[i] / fs))
        elif fs_list[i] < fs:
            data = resample(data, int(data.shape[-1] * (fs / fs_list[i])), axis=1)


        seq_len = data.shape[-1] # get the length of the ecg sequence
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)

        # Polarity check, per selected channel
        for ch_idx in polarity_check:
            try:
                # Get BioSPPy ECG object, using specified channel
                ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs_list[i], show=False)

                # Get rpeaks and beat templates
                rpeaks = ecg_object['rpeaks']
                templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

                # Polarity check (based on extremes of median templates)
                templates_min = np.min(np.median(templates, axis=1)) 
                templates_max = np.max(np.median(templates, axis=1))

                if np.abs(templates_min) > np.abs(templates_max):
                    # Flip polarity
                    data[ch_idx] *= -1
                    templates *= -1
            except:
                continue

        data = normalize(data)
        lbl = labels_filtered[i]

        # Add just enough padding to allow window
        pad = np.abs(np.min(seq_len - window, 0))
        if pad > 0:
            data = np.pad(data, ((0,0),(0,pad+1)))
            seq_len = data.shape[-1] # get the new length of the ecg sequence

        starts = np.random.randint(seq_len - window + 1, size=nb_windows) # get start indices of ecg segment        
        ecg_segs.append(np.array([data[:,start:start+window] for start in starts]))
    return ecg_segs
# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

#convert above to function
def get_features(ecg_data_filtered, fs_list, filter_bandwidth, ch_idx, error_indices,fs=500):
    features = []
    error_indices = []
    for i in range(len(ecg_data_filtered)):
        #print("Processing:",i,"/",len(ecg_data_filtered))
        if fs_list[i] > fs:
            recording = decimate(ecg_data_filtered[i], int(fs_list[i] / fs))
        elif fs_list[i] < fs:
            recording = resample(ecg_data_filtered[i], int(ecg_data_filtered[i].shape[-1] * (fs / fs_list[i])), axis=1)
        else:
            recording = ecg_data_filtered[i]
        if i in error_indices:
            continue
        if i%500 == 0:
            print("Processing:",i,"/",len(ecg_data_filtered))
        ecg_features = Features(
            data=recording[ch_idx],
            fs=fs_list[i],
            feature_groups=['full_waveform_statistics', 'heart_rate_variability_statistics', 'template_statistics']
        )
        try:
            ecg_features.calculate_features(
                filter_bandwidth=[3, 45], show=False,
                channel=0, normalize=True, polarity_check=True,
                template_before=0.25, template_after=0.4
            )
            feats = ecg_features.get_features()
            features.append(feats)
        except:
            # remove the data from the ecg_data_filtered list and labels_filtered list
            print("Error in processing:",i)
            #save the indices of the error
            error_indices.append(i)
            continue
    return features, error_indices


def start_log(loc, fold):
    if not (loc/f'log_fold_{fold}.csv').exists():
        with open(loc/f'log_fold_{fold}.csv', 'w') as f:
            f.write('epoch, trn_loss, trn_auroc, val_loss, val_auroc\n')

def create_experiment_directory(output_directory):
    results_loc = Path(output_directory)/'saved_models'
    #results_loc.mkdir(exist_ok=True)
 
    results_loc = results_loc/model_name
    #results_loc.mkdir(exist_ok=True)

    sw = SummaryWriter(log_dir=results_loc)
    return results_loc, sw   

def load_best_model(model_loc, model):
    checkpoint = torch.load(model_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loading best model: best_loss', checkpoint['best_loss'], 'best_auroc', checkpoint['best_auroc'], 'at epoch', checkpoint['epoch'])
    return model

def create_fold_dir(results_loc, fold):
    fold_loc = results_loc/f'fold_{fold}'
    fold_loc.mkdir(exist_ok=True)
    return fold_loc

import torch
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    def __init__(self, ecg_segs, feats_normalized, labels,age,gender):
        self.ecg_segs = ecg_segs
        self.feats_normalized = feats_normalized
        self.labels = labels
        self.age = age
        self.gender = gender

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg_segs[idx], self.feats_normalized[idx], self.labels[idx], self.age[idx], self.gender[idx]



if __name__ == '__main__':
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    # Create a new directory in output_dir with the current time
    output_dir = f"/work/home/sk87dewu/output_dir/{time_str}"
    os.makedirs(output_dir, exist_ok=True)

    
    gender_1, age_1, labels_1, ecg_filenames_1, ecg_data_1, header_data_1 = import_key_data("/work/home/sk87dewu/data/cpsc_2018")
    gender_2, age_2, labels_2, ecg_filenames_2, ecg_data_2, header_data_2 = import_key_data("/work/home/sk87dewu/data/cpsc_2018_extra")
    gender_3, age_3, labels_3, ecg_filenames_3, ecg_data_3, header_data_3 = import_key_data("/work/home/sk87dewu/data/ptb")
    gender_4, age_4, labels_4, ecg_filenames_4, ecg_data_4, header_data_4 = import_key_data("/work/home/sk87dewu/data/ptb-xl")
    gender_5, age_5, labels_5, ecg_filenames_5, ecg_data_5, header_data_5 = import_key_data("/work/home/sk87dewu/data/st_petersburg_incart")
    gender_6, age_6, labels_6, ecg_filenames_6, ecg_data_6, header_data_6 = import_key_data("/work/home/sk87dewu/data/georgia")

    features_df = pd.read_csv(r"/work/home/sk87dewu/output_dir/features_prna_merged_all.csv")

    #merge all data into one list
    ecg_data = ecg_data_1 + ecg_data_2 + ecg_data_3 + ecg_data_4 + ecg_data_5 + ecg_data_6
    header_data = header_data_1 + header_data_2 + header_data_3 + header_data_4 + header_data_5 + header_data_6
    labels = labels_1 + labels_2 + labels_3 + labels_4 + labels_5 + labels_6
    gender = gender_1 + gender_2 + gender_3 + gender_4 + gender_5 + gender_6
    age = age_1 + age_2 + age_3 + age_4 + age_5 + age_6

    fs_list = []
    for i in range(len(ecg_data)):
        fs_list.append(int(header_data[i][0].split(' ')[2]))

    
    

    

    # fetch ids from header_data
    ids = []
    for i in range(len(ecg_data)):
        ids.append(header_data[i][0].split(' ')[0])
    fs_list_dict = dict(zip(ids,fs_list))
    
    # create a dict of id as key and ecg_data as value
    ecg_data_dict = dict(zip(ids, ecg_data))
    age = [float(a) for a in age]
    # taje mean of age_list and replace nan with mean value
    mean_age = np.nanmean(age)
    age = [mean_age if np.isnan(x) else x for x in age]
    age_dict = dict(zip(ids, age))

    gender = [gen.strip() for gen in gender]
    #change gender list to binary 0 for Female and 1 for Male
    gender = [1 if x=="Male" else 0 for x in gender]
    gender = [float(g) for g in gender]
    

    gender_dict = dict(zip(ids,gender))
    # get ages in order of features_df based on id column and id in age_dict


    ages = []
    genders = []
    fs_list = []
    for i in range(len(features_df)):
        id = features_df.loc[i,"id"]
        ages.append(age_dict[id])
        genders.append(gender_dict[id])
        fs_list.append(fs_list_dict[id])


    # normalize ages by mean and std
    ages = (ages - np.mean(ages)) / np.std(ages)
  
    nb_windows=20
    window = 2500
    deepfeat_sz = 64
    dropout_rate = 0.2
    #fs = 500
    filter_bandwidth = [3, 45]
    polarity_check = []

    # Transformer parameters
    d_model = 256   # embedding size
    nhead = 8       # number of heads
    d_ff = 2048     # feed forward layer size
    num_layers = 8  # number of encoding layers

    ch_idx = 1
    nb_demo = 2
    nb_feats = 20
    patience_count = 0

    model_name = 'ctn'
    feature_names = list(np.load(r"/work/home/sk87dewu/scripts/top_feats.npy",allow_pickle=True))
    feature_names.remove('full_waveform_duration')
    feature_names.remove('Age')
    feature_names.remove('Gender_Male')
    top_feats = features_df[feature_names[:nb_feats]].values
   # First, convert any infs to nans
    top_feats[np.isinf(top_feats)] = np.nan
       # Replace NaNs with feature means
    feat_means = np.nanmean(top_feats, axis=0)
    feat_stds = np.nanstd(top_feats, axis=0)
   # Replace NaNs with feature means
    mask = np.isnan(top_feats)
    for i in range(top_feats.shape[1]):
        top_feats[mask[:, i], i] = feat_means[i]
       # Normalize wide features
    #top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
       # Normalize wide features

    feats_normalized = (top_feats - feat_means) / feat_stds
    if not len(feats_normalized):
        feats_normalized = np.zeros(nb_feats)[None]
    oe = OneHotEncoder()
    
    labels_filtered = oe.fit_transform(features_df['merged_labels'].values.reshape(-1,1)).toarray()
    
    np.save(r"/work/home/sk87dewu/output_dir/one_hot_order_full.npy",oe.categories_[0])

    
    classes = 27
    
    ecg_segs = create_ecg_segs(ecg_data_dict, labels_filtered, window, nb_windows, filter_bandwidth, polarity_check,fs_list,features_df)

    # filter header_data based on features_df id column as there might be duplicates and not based on index

    header_data_filtered = []
    for i in range(len(features_df)):
        id = features_df.loc[i,"id"]
        header_data_filtered.append(header_data[ids.index(id)])

    #get ages from header data in order of features_df based on id column and header data id value
    
    
    assert len(ages) == len(features_df)
    assert len(ecg_segs) == len(features_df)
    assert len(feats_normalized) == len(features_df)
    assert len(labels_filtered) == len(features_df)
    assert len(header_data_filtered) == len(features_df)
    assert len(genders) == len(features_df)
    # Create the dataset
    dataset = ECGDataset(ecg_segs, feats_normalized, labels_filtered, ages,genders)

    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import Subset

   # Convert labels to numpy array for StratifiedShuffleSplit
   #labels_np = labels_filtered.numpy()

   # Create the stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    le = LabelEncoder()
    labels_filtered = le.fit_transform(list(features_df["merged_labels"]))
    
   # Get the indices for the temporary training set and the test set
    for temp_train_index, test_index in sss.split(torch.zeros(len(labels_filtered)), labels_filtered):
        temp_train_dataset = Subset(dataset, temp_train_index)
        test_dataset = Subset(dataset, test_index)
        
    # Create a temporary labels array for the second split
    temp_train_labels = labels_filtered[temp_train_index]
    
    # Create another stratified split for the temporary training set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)


   # Get the indices for the final training set and the validation set
    for train_index, val_index in sss.split(torch.zeros(len(temp_train_labels)), temp_train_labels):
        temp_train_index = train_index
        val_index = val_index
        train_dataset = Subset(dataset, temp_train_index)
        val_dataset = Subset(dataset, val_index)
    

   # Create the dataloaders
    trnloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=3)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=3)
    tstloader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=3)
#    for i, batch in enumerate(trnloader):
#      print(f"Batch {i+1}:")
#      for j, item in enumerate(batch):
#          if isinstance(item, list):
#              print(f"Item {j+1} length: {len(item)}")
#          else:
#              print(f"Item {j+1} shape: {item.shape}")
    

    
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    print(f'Number of params: {sum([p.data.nelement() for p in model.parameters()])}')

    optimizer = NoamOpt(d_model, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #convert ecg_segs, feats_normalized, labels_filtered to dataloader
    tst_fold = 0
    # Create dir structure and init logs
    results_loc, sw = create_experiment_directory(output_dir)
    fold_loc = create_fold_dir(results_loc, tst_fold)
    start_log(fold_loc, tst_fold)
    thrs = np.array([0.5])

    if do_train:
        for epoch in range(10):
            trn_loss, trn_auroc = train(epoch, model, trnloader, optimizer)
            val_loss, val_auroc = validate(epoch, model, valloader, optimizer, fold_loc)
            write_log(fold_loc, tst_fold, epoch, trn_loss, trn_auroc, val_loss, val_auroc)
            print(f'Train - loss: {trn_loss}, auroc: {trn_auroc}')
            print(f'Valid - loss: {val_loss}, auroc: {val_auroc}')
            
            sw.add_scalar(f'{tst_fold}/trn/loss', trn_loss, epoch)
            sw.add_scalar(f'{tst_fold}/trn/auroc', trn_auroc, epoch)
            sw.add_scalar(f'{tst_fold}/val/loss', val_loss, epoch)
            sw.add_scalar(f'{tst_fold}/val/auroc', val_auroc, epoch)

            # Early stopping
            if patience_count >= patience:
                print(f'Early stopping invoked at epoch, #{epoch}')
                break
        
    # Training done, choose threshold...
    model = load_best_model(str(f'{fold_loc}/{model_name}.tar'), model)
    probs, lbls = get_probs(model, tstloader)
    
    print("test_probs: ",probs)
    print("test_labels: ",lbls)    
    
    #preds = (probs > thrs).astype(int)
    preds = np.array(probs)
    lbls = np.array(lbls)
    true_labels = np.argmax(lbls, axis=1)
    predictions = np.argmax(preds, axis=1)
    #print(thrs)
    # print classification report getting labels from valloader


    
    clf_report = pd.DataFrame(classification_report(true_labels, predictions, output_dict=True)).T
    print(clf_report)
    clf_report.to_csv(fold_loc/'classification_report.csv')
