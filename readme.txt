Sau khi đã cài đặt môi trường và các thư viện ta có thể tiến hành chạy code

Với thuật toán ANN
Đầu tiên ta sẽ trích xuất đặc trưng âm thanh của tập train

DÀNH CHO KỊCH BẢN 1

LƯU Ý:NÊN KIỂM TRA TẬP TRAIN VÀ TEST TRƯỚC KHI CHẠY. Nếu trong tập train và tập test còn file desktop.ini thì hãy xóa file này trước khi chạy

Đối với tỷ lệ lấy mẫu sr=16000 thì ta chạy

python train_feature_16.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 100 --nfilt 10
python train_feature_16.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 110 --nfilt 11
python train_feature_16.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 120 --nfilt 12
python train_feature_16.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 130 --nfilt 13
python train_feature_16.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 140 --nfilt 14


Sau đó ta sẽ chạy tiếp 

python find_song_16.py --train_folder train200/ --test_folder test30/  --n 1 --f 100 --nfilt 10
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 5 --f 100 --nfilt 10
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 10 --f 100 --nfilt 10

python find_song_16.py --train_folder train200/ --test_folder test30/  --n 1 --f 110 --nfilt 11
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 5 --f 110 --nfilt 11
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 10 --f 110 --nfilt 11

python find_song_16.py --train_folder train200/ --test_folder test30/  --n 1 --f 120 --nfilt 12
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 5 --f 120 --nfilt 12
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 10 --f 120 --nfilt 12

python find_song_16.py --train_folder train200/ --test_folder test30/  --n 1 --f 130 --nfilt 13
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 5 --f 130 --nfilt 13
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 10 --f 130 --nfilt 13

python find_song_16.py --train_folder train200/ --test_folder test30/  --n 1 --f 140 --nfilt 14
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 5 --f 140 --nfilt 14
python find_song_16.py --train_folder train200/ --test_folder test30/  --n 10 --f 140 --nfilt 14


SAU KHI TA CHẠY HẾT TẤT CẢ CÁC LỆNH TRÊN TA CẦN COPY KẾT QUẢ VÀO 1 FOLDER KHÁC RỒI MỚI THỰC HIỆN TIẾP CÁC LỆNH BÊNH DƯỚI


Đối với tỷ lệ lấy mẫu sr=40000 thì ta chạy

python train_feature_40.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 100 --nfilt 10
python train_feature_40.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 110 --nfilt 11
python train_feature_40.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 120 --nfilt 12
python train_feature_40.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 130 --nfilt 13
python train_feature_40.py --train_folder train200/ --test_folder test1/DangDo  --n 1 --f 140 --nfilt 14


Sau đó ta sẽ chạy tiếp 

python find_song_40.py --train_folder train200/ --test_folder test30/  --n 1 --f 100 --nfilt 10
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 5 --f 100 --nfilt 10
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 10 --f 100 --nfilt 10

python find_song_40.py --train_folder train200/ --test_folder test30/  --n 1 --f 110 --nfilt 11
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 5 --f 110 --nfilt 11
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 10 --f 110 --nfilt 11

python find_song_40.py --train_folder train200/ --test_folder test30/  --n 1 --f 120 --nfilt 12
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 5 --f 120 --nfilt 12
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 10 --f 120 --nfilt 12

python find_song_40.py --train_folder train200/ --test_folder test30/  --n 1 --f 130 --nfilt 13
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 5 --f 130 --nfilt 13
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 10 --f 130 --nfilt 13

python find_song_40.py --train_folder train200/ --test_folder test30/  --n 1 --f 140 --nfilt 14
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 5 --f 140 --nfilt 14
python find_song_40.py --train_folder train200/ --test_folder test30/  --n 10 --f 140 --nfilt 14

TA CŨNG COPY CÁC KẾT QUẢ ĐẠT ĐƯỢC VÀO FOLDER KHÁC TRƯỚC KHI CHẠY VỚI CÁC test_folder khác

TA CÓ THỂ THAY test30 bằng 30s, 60s, 120s để chạy các file find_song_16.py và find_song_40.py

LƯU Ý 2: NẾU ĐÃ CHẠY FILE train_feature_40.py THÌ FILE TÌM KIẾM CŨNG PHẢI CHẠY find_song_40.py 
CÒN NẾU CHẠY FILE train_feature_16.py THÌ FILE TÌM KIẾM CŨNG PHẢI CHẠY FILE find_song_16.py


ĐỐI VỚI KỊCH BẢN 2

Đầu tiên ta phải chạy 

python train_feature_16.py --train_folder train100/ --test_folder test1/DangDo  --n 1 --f 100 --nfilt 10
python train_feature_16.py --train_folder train100/ --test_folder test1/DangDo  --n 1 --f 110 --nfilt 11
python train_feature_16.py --train_folder train100/ --test_folder test1/DangDo  --n 1 --f 120 --nfilt 12
python train_feature_16.py --train_folder train100/ --test_folder test1/DangDo  --n 1 --f 130 --nfilt 13
python train_feature_16.py --train_folder train100/ --test_folder test1/DangDo  --n 1 --f 140 --nfilt 14


Kế tiếp ta chạy

python find_song_16.py --train_folder train100/ --test_folder test100/  --n 1 --f 100 --nfilt 10
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 5 --f 100 --nfilt 10
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 10 --f 100 --nfilt 10

python find_song_16.py --train_folder train100/ --test_folder test100/  --n 1 --f 110 --nfilt 11
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 5 --f 110 --nfilt 11
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 10 --f 110 --nfilt 11

python find_song_16.py --train_folder train100/ --test_folder test100/  --n 1 --f 120 --nfilt 12
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 5 --f 120 --nfilt 12
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 10 --f 120 --nfilt 12

python find_song_16.py --train_folder train100/ --test_folder test100/  --n 1 --f 130 --nfilt 13
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 5 --f 130 --nfilt 13
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 10 --f 130 --nfilt 13

python find_song_16.py --train_folder train100/ --test_folder test100/  --n 1 --f 140 --nfilt 14
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 5 --f 140 --nfilt 14
python find_song_16.py --train_folder train100/ --test_folder test100/  --n 10 --f 140 --nfilt 14


KỊCH BẢN 3

Đầu tiên là chuyển âm thanh thành ảnh phổ

python preprocessing.py

Sau đó chạy các model để huấn luyện với ảnh phổ.

python cv-run_save_vgg16.py -t train -r test -e 100 -c 100 -k 0 --mod vgg
python cv-run_save_vgg16.py -t train -r test -e 100 -c 100 -k 0 --mod cnn1
python cv-run_save_vgg16.py -t train -r test -e 100 -c 100 -k 0 --mod cnn2
python cv-run_save_vgg16.py -t train -r test -e 100 -c 100 -k 0 --mod cnn3
python cv-run_save_vgg16.py -t train -r test -c 100 -e 100 -i 224 --mod efficient --learning_rate 0.00005
python cv-run_save_vgg16.py -t train -r test -e 100 -c 100 -k 0 --mod fc

Các kết quả huấn luyện sẽ nằm trong folder save_models


