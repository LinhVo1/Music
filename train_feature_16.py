import annoy
from annoy import AnnoyIndex
import librosa 
import os 
import glob
import soundfile
from tqdm import tqdm 
import numpy as np
import sys
from python_speech_features import mfcc, fbank, logfbank
import pickle
import time
import argparse
import datetime
from datetime import date
from  collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', help='kho')
parser.add_argument('--test_folder', help='test folder')
parser.add_argument('--n', type=int, default=1, help='number of songs returned(1,5,10)')
parser.add_argument('--f', type=int, default=1, help='vector size(100,110,120,130)')
parser.add_argument('--nfilt', type=int, default=1, help='length of each MFCC(10,11,12,13)')
opt = parser.parse_args()

def scan_music(test_folder, filename):
    try:
        song = os.path.join(test_folder, filename + '.wav')
        y, sr = librosa.load(song,sr=16000)
        print(y,sr)
    except:
        song = os.path.join(test_folder, filename + '.mp3')
        y, sr = librosa.load(song,sr=16000)
        print(y,sr)
        
def extract_features(y, sr=16000, nfilt=10,winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep =winsteps)
        return feat
    except:
        raise Exception("Extraction features error")

def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)),mode='constant')
    return crop_feat


def find(train_folder,features=[],songs=[]):
    for song in tqdm(os.listdir(train_folder)):
        song = os.path.join(train_folder,song)
        y, sr = librosa.load(song, sr=16000)
        feat = extract_features(y, sr, opt.nfilt)
        for i in range(0, feat.shape[0], 5):
            features.append(crop_feature(feat, i, nb_step=10,maxlen=10*opt.nfilt))
            songs.append(song)           
features = []
songs = []

find(opt.train_folder, features, songs)

features_pk='features_'+str(opt.f)+'_'+str(opt.nfilt)+'.pk'
songs_pk='songs_'+str(opt.f)+'_'+str(opt.nfilt)+'.pk'
pickle.dump(features,open(features_pk, 'wb'))
pickle.dump(songs, open(songs_pk, 'wb'))

t = AnnoyIndex(opt.f)
len_features=len(features)
for i in range(len_features):
    v = features[i]
    t.add_item(i,v) 

start = time.time()
t.build(opt.f)
end = time.time()-start
music_ann='music_'+str(opt.f)+'_'+str(opt.nfilt)+'.ann'
t.save(music_ann)


try:
    song = os.path.join(opt.test_folder + '.wav')
    y, sr = librosa.load(song, sr=16000)
    feat = extract_features(y)
except:
     song = os.path.join(opt.test_folder +'.mp3')
     y, sr = librosa.load(song,sr=16000)
     feat = extract_features(y)
    
file_results= str(datetime.date.today()) + "-ketqua-" + str(opt.n)+'_'+str(opt.f)+'_'+str(opt.nfilt)+'.txt'
file = open(file_results,'w')         
results = []
u = AnnoyIndex(opt.f, metric='angular')
u.load(music_ann)

for i in range(0, feat.shape[0], 5):
    crop_feat = crop_feature(feat, i , nb_step=10,maxlen=10*opt.nfilt)
    result = u.get_nns_by_vector(crop_feat, opt.n)
    result_songs = [songs[k] for k in result]
    results.append(result_songs)   
results = np.array(results).flatten()

most_song = Counter(results)
most_song.most_common()

file.writelines('Độ dài của mỗi MFCC(nfilt): '+ str(opt.nfilt) + '\n')
file.writelines('Số chiều vector(F): '+ str(opt.f) + '\n')
length = len(opt.train_folder)
file.writelines('Top '+ str(opt.n) +' bài hát gần giống nhất với bài hát: '+opt.test_folder+'\n')

for i in range(0,opt.n):
    print(most_song.most_common()[i][0][length:]+'->'+str(most_song.most_common()[i][1]))
    file.writelines(most_song.most_common()[i][0][length:]+'->'+str(most_song.most_common()[i][1])+'\n')   

file.writelines('Thời gian tiền xử lý (giây): '+str(time.process_time()) + '\n')

file.writelines('Thời gian tạo cây chỉ mục (giây):  %.5f' %end + '\n')
file.close()
# python train_feature_16.py --train_folder train/ --test_folder test1/DangDo  --n 1 --f 100 --nfilt 10