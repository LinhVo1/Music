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
from  collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', help='kho')
parser.add_argument('--test_folder', help='test folder')
parser.add_argument('--n', type=int, default=1, help='number of songs returned(1,5,10)')
parser.add_argument('--f', type=int, default=1, help='vector size(100,110,120,130)')
parser.add_argument('--nfilt', type=int, default=1, help='length of each MFCC(10,11,12,13)')
opt = parser.parse_args()

u = AnnoyIndex(opt.f, metric='angular')
music_ann='music_'+str(opt.f)+'_'+str(opt.nfilt)+'.ann'
u.load(music_ann)

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

features_pk='features_'+str(opt.f)+'_'+str(opt.nfilt)+'.pk'
songs_pk='songs_'+str(opt.f)+'_'+str(opt.nfilt)+'.pk'
features=pickle.load(open(features_pk, 'rb'))
songs=pickle.load(open(songs_pk, 'rb'))

file_results=str(datetime.date.today()) +'_ketqua_top_'+str(opt.n)+'_'+str(opt.f)+'_'+str(opt.nfilt)+'.txt'
file = open(file_results,'w')
file.writelines('Độ dài của mỗi MFCC(nfilt): '+ str(opt.nfilt) + '\n')
file.writelines('Số chiều vector(F): '+ str(opt.f) + '\n')

   

stt=1
for namesong in tqdm(os.listdir(opt.test_folder)):
    start=time.time()
    namesong = os.path.join(opt.test_folder,namesong)
    y, sr = librosa.load(namesong, sr=16000)
    feat = extract_features(y, sr, opt.nfilt) 
    results = []
    for i in range(0, feat.shape[0], 5):
        crop_feat = crop_feature(feat, i , nb_step=10,maxlen=10*opt.nfilt)
        result = u.get_nns_by_vector(crop_feat, opt.n)
        result_songs = [songs[k] for k in result]
        results.append(result_songs)
        
    results = np.array(results).flatten()
    end=time.time()-start
    
    most_song = Counter(results)
    most_song.most_common()
   
    file.writelines( str(stt)+': Top '+ str(opt.n) +' bài hát gần giống nhất so với bài hát: '+namesong+ '\n')
    stt+=1
    length = len(opt.test_folder)
    file_name=namesong[length+1:]
    for i in range(0,opt.n):
        file.writelines(most_song.most_common()[i][0][length:]+'->'+str(most_song.most_common()[i][1])+'\n')
    file.writelines('\n')
    file.writelines('Thời gian tìm là (giây):  %.5f' %end + '\n')
    file.writelines('-------------------------------------------')
    file.writelines('\n\n\n')
file.writelines('Thời gian tiền xử lý (giây): '+str(time.process_time()) + '\n')
file.close()

# python find_song_16.py --train_folder train/ --test_folder test/  --n 1 --f 100 --nfilt 10

