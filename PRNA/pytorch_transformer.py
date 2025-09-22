#from train_12ECG_classifier import train_12ECG_classifier
from feats.features import *
import wfdb
from scipy.io import loadmat
import os
import numpy as np
import neurokit2 as nk
#from model import CTN
#from utils import *
#from train_12ECG_classifier import train,validate
#from optimizer import NoamOpt
from tensorboardX import SummaryWriter
# stop warningd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report


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
    # for subdir, dirs, files in sorted(os.walk(path)):
    #     for dir in dirs:
    for subdir2, dirs2, files2 in sorted(os.walk(path)):
        print("Reading:", subdir2)
        for filename in files2:
            filepath = subdir2 + os.sep + filename
            if filepath.endswith(".mat"):
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
def create_ecg_segs(ecg_data_filtered, labels_filtered, window, nb_windows, filter_bandwidth, polarity_check):
    ecg_segs = []
    for i in range(len(ecg_data_filtered)):
        data = ecg_data_filtered[i]
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

#convert above to function
def get_features(ecg_data_filtered, fs, filter_bandwidth, ch_idx, error_indices):
    features = []
    error_indices = []
    for i in range(len(ecg_data_filtered)):
        if i in error_indices:
            continue
        if i%500 == 0:
            print("Processing:",i,"/",len(ecg_data_filtered))
        ecg_features = Features(
            data=ecg_data_filtered[i][ch_idx],
            fs=fs,
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

if __name__ == '__main__':
    gender_1, age_1, labels_1, ecg_filenames_1, ecg_data_1, header_data_1 = import_key_data(r"C:\Users\adity\FL_practice\thesis_data\classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2\training\cpsc_2018")
    gender_2, age_2, labels_2, ecg_filenames_2, ecg_data_2, header_data_2 = import_key_data("/work/home/sk87dewu/data/cpsc_2018_extra")
    gender_3, age_3, labels_3, ecg_filenames_3, ecg_data_3, header_data_3 = import_key_data("/work/home/sk87dewu/data/ptb")
    gender_4, age_4, labels_4, ecg_filenames_4, ecg_data_4, header_data_4 = import_key_data("/work/home/sk87dewu/data/ptb-xl")
    gender_5, age_5, labels_5, ecg_filenames_5, ecg_data_5, header_data_5 = import_key_data("/work/home/sk87dewu/data/st_petersburg_incart")

    # merge all data into one list
    ecg_data = ecg_data_1 + ecg_data_2 + ecg_data_3 + ecg_data_4 + ecg_data_5
    header_data = header_data_1 + header_data_2 + header_data_3 + header_data_4 + header_data_5
    labels = labels_1 + labels_2 + labels_3 + labels_4 + labels_5
    gender = gender_1 + gender_2 + gender_3 + gender_4 + gender_5
    age = age_1 + age_2 + age_3 + age_4 + age_5

    labels = [label.strip() for label in labels]

    dx_map_df = pd.read_csv(r'/work/home/sk87dewu/scripts/dx_mapping_scored.csv',sep=',')
    
    labels_filtered, indices = filter_codes(labels, dx_map_df)
    ecg_data_filtered = [ecg_data[i] for i in indices]
    header_data_filtered = [header_data[i] for i in indices]
    fs_list = []
    for i in range(len(ecg_data_filtered)):
        fs_list.append(int(header_data_filtered[i][0].split(' ')[2]))
    nb_windows=30
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

    # Get features
    features, error_indices = get_features(ecg_data_filtered, fs_list, filter_bandwidth, ch_idx, fs_list)

    # get ids from header_data from filtered indices
    ids = []
    for i in indices:
        
        ids.append(header_data[i][0].split(' ')[0])

    # remove the error indices from the ids
    # combine list of features into a single dataframe
    features_df = pd.concat(features)
    features_df = features_df.reset_index(drop=True)

    ids = [ids[i] for i in range(len(ids)) if i not in error_indices]
    labels_filtered = [labels_filtered[i] for i in range(len(labels_filtered)) if i not in error_indices]
    features_df["id"] = ids
    features_df['labels'] = labels_filtered
    features_df['label_dx'] = features_df['label'].map(dx_map_df.set_index('SNOMED CT Code')['Abbreviation'])
    features_df.to_csv("features.csv",index=False)
    #model_name = 'ctn'
    feature_names = list(np.load('top_feats.npy'))
    feature_names.remove('full_waveform_duration')
    feature_names.remove('Age')
    feature_names.remove('Gender_Male')
    top_feats = features_df[feature_names[:250]].values
    top_feats[np.isinf(top_feats)] = np.nan
    # Replace NaNs with feature means
    feat_means = np.nanmean(top_feats, axis=0)
    feat_stds = np.nanstd(top_feats, axis=0)
    # Replace NaNs with feature means
    mask = np.isnan(top_feats)
    for i in range(top_feats.shape[1]):
        top_feats[mask[:, i], i] = feat_means[i]

    X = top_feats
    y = features_df['label_dx']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # define the pipeline
    pipeline = make_pipeline( RandomOverSampler(sampling_strategy='auto', random_state=42))#SMOTE(sampling_strategy='auto', k_neighbors=1),

    # fit the pipeline on the training data
    X_train, y_train = pipeline.fit_resample(X_train, y_train)

    # define the model
    clf = CatBoostClassifier(iterations=10000, depth=10, learning_rate=0.1, loss_function='MultiClass', verbose=True, task_type='GPU', devices='0:3')
    # fit the model
    clf.fit(X_train, y_train)
    # make predictions
    y_pred = clf.predict(X_test)

    # print the classification report and save as df
    clf_report = pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose()
    print(pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose())
    clf_report.to_csv("/work/home/sk87dewu/output_dir/clf_report.csv",index=False)


    #save features in order of most important features
    feature_importances = clf.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    # save the top 250 features
    top_feats = np.array(feature_names)[indices[:250]]
    np.save('/work/home/sk87dewu/output_dir/top_feats.npy',top_feats)

    #save the model to output_dir
    clf.save_model('/work/home/sk87dewu/output_dir/catboost_model.cbm')
    print("Model saved to output_dir")
