import pickle
from flask import Flask, render_template, request
import os
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
from annoy import AnnoyIndex
from collections import Counter
import array as arr

f=int(120)
nfilt=int(12)
n=int(5)
folder_train = 'train200'
folder_test = 'test_find'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "test_find"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    
    if request.method == 'POST':
        try:
            
            song_find = request.files['file']
            m = request.values['n']
            if m:
                n = int(m)
            
            if song_find:
                 
                print(song_find)
                print(app.config['UPLOAD_FOLDER'])
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], song_find.filename)
                print("Save = ", path_to_save)
                song_find.save(path_to_save)
                
                
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

                songs_pk='songs_120_12.pk'
                songs=pickle.load(open(songs_pk, 'rb'))


                for filename in tqdm(os.listdir(folder_test)):
                    filename = os.path.join(folder_test, filename)
                    y, sr = librosa.load(filename, sr=16000)
                    feat = extract_features(y, sr, nfilt)
                    
                    results = []
                    
                    u = AnnoyIndex(120)
                    music_ann='music_120_12.ann'
                    
                    u.load(music_ann)  
                   
                    for i in range(0, feat.shape[0], n):
                        crop_feat = crop_feature(feat, i , nb_step=10,maxlen=10*nfilt)
                        result = u.get_nns_by_vector(crop_feat, n)
                        result_songs = [songs[k] for k in result]
                        results.append(result_songs)
                        
                    results = np.array(results).flatten()

                    most_song = Counter(results)
                    most_song.most_common()
                    
                    length_folder_test = len(folder_train)
                    kq=[]
                    for i in range(0,n):
                        kq.append(most_song.most_common()[i][0][length_folder_test-3:]) 
                        print(most_song.most_common()[i][0][length_folder_test-3:])
                    print(len(kq))
                    return render_template("index.html", music_name = kq)
            
            else:
                
                return render_template('index.html', msg='Hãy chọn file để tải lên')
        except Exception as ex:
            
            print(ex)
            return render_template('index.html', msg=' Không xác định được bài hát ')
    else:
        
        return render_template('index.html')
       
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)