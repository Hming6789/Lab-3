## Phân tích kết quả đo được

### Dữ liệu và đặc trưng

40 WAV eSpeak, 30 train/10 test, mọi header và metadata khớp. Không trùng byte hoặc PCM giữa train/test, không có clipping theo ngưỡng kiểm tra. Chưa xác nhận bằng nghe. Hash khác nhau không chứng minh độc lập thống kê giữa các WAV cùng bộ TTS.

Hai ví dụ Viterbi `mo_07` có 25 frame, `xuong_07` có 28 frame sau endpoint detection. MFCC biểu diễn khác biệt phổ; delta/delta² nêu diễn biến quanh chuyển đoạn trên heatmap. Mỗi đặc trưng là T×39, CMVN trong từng utterance.

### HMM và hội tụ

Mô hình chính đạt 10/10 test sạch. Không có cặp nhầm lẫn trên tập sạch. Score gap/frame nhỏ nhất vẫn lớn hơn 0.5, không có WAV acoustic gần nhau theo tiêu chí đặt trước. Các likelihood là mật độ của feature liên tục, không phải xác suất dự đoán được hiệu chuẩn.

- `mo`: 16 iteration, dừng `decreased`, delta cuối -0.001358; A hợp lệ, π cố định.
- `dong`: 6 iteration, dừng `decreased`, delta cuối -0.001076; A hợp lệ, π cố định.
- `len`: 12 iteration, dừng `decreased`, delta cuối -0.013882; A hợp lệ, π cố định.
- `xuong`: 6 iteration, dừng `tolerance`, delta cuối 0.000864; A hợp lệ, π cố định.
- `dung`: 6 iteration, dừng `tolerance`, delta cuối 0.000015; A hợp lệ, π cố định.

Likelihood giảm nhẹ ở một số model: không gọi chúng hội tụ tốt chỉ từ cờ `monitor.converged`. Prior/regularization và tối ưu số học có thể làm mục tiêu likelihood monitor giảm nhẹ; mọi cảnh báo được giữ, không che đi. Cần thêm dữ liệu/validation và kiểm tra nhiều khởi tạo nếu cải tiến hệ thống.

### E1 đến E3

- `E1_3`: accuracy 100%; 7.6 vòng trung bình; 1/5 model đạt tolerance; train/test true-label score/frame -29.57/-37.24.
- `baseline`: accuracy 100%; 9.2 vòng trung bình; 2/5 model đạt tolerance; train/test true-label score/frame -16.95/-29.72.
- `E1_7`: accuracy 100%; 6.4 vòng trung bình; 0/5 model đạt tolerance; train/test true-label score/frame -10.16/-25.00.
- `E2_13`: accuracy 100%; 7.8 vòng trung bình; 0/5 model đạt tolerance; train/test true-label score/frame -3.78/-5.35.
- `E3_full`: accuracy 100%; 3.4 vòng trung bình; 4/5 model đạt tolerance; train/test true-label score/frame 52.77/-561.79.

E1: tăng state 3→5→7 làm train score tăng, nhưng accuracy sạch đều bằng 100%, chưa có bằng chứng cần nhiều state hơn. 7 states không model nào đạt tolerance dương không giảm, nên không chọn 7 chỉ vì train likelihood cao. Giữ 5 states theo PDF đã chốt trước.

E2: cả 13 và 39 chiều đều 100%, recall cả 5 lớp đều 1.0; chưa thấy lớp nào được delta cải thiện trên tập test sạch này. Không so trị tuyệt đối likelihood giữa hai số chiều.

E3: full covariance có chênh lệch train/test rất lớn (52.77 so với −561.79/frame), trong khi diag −16.95/−29.72. Đây là dấu hiệu phù hợp với overfit và covariance thiếu ổn định. Có regularization nên vẫn chấm hữu hạn; accuracy 100% chỉ nói thứ hạng trong 5 lớp còn đúng, không chứng minh full tổng quát hóa tốt. Số frame mỗi state nhỏ so với 780 tham số covariance; frame cũng tương quan, nên full chỉ là stress test có prior.

### E4 và tốc độ nói

- HMM39: 100%, chuẩn bị 0.940906 s, dự đoán 20.032 ms/file; 6 template/train utterance mỗi lớp.
- DTW13: 100%, chuẩn bị 0.004724 s, dự đoán 26.557 ms/file; 6 template/train utterance mỗi lớp.
- DTW39: 100%, chuẩn bị 0.004377 s, dự đoán 25.384 ms/file; 6 template/train utterance mỗi lớp.

Không có file sạch mà HMM và DTW13 khác quyết định. Đối chiếu có kiểm soát tốc độ với DTW39 (cùng 39D):

- Rate 0.8: HMM sai 2/10; DTW39 đúng 10/10. Các file HMM sai: dong_07.wav: dong → xuong; dong_08.wav: dong → xuong.
- Rate 1.2: HMM sai 1/10; DTW39 đúng 10/10. Các file HMM sai: dong_07.wav: dong → xuong.

DTW warping linh hoạt hơn trên các biến đổi tốc độ nhân tạo này; HMM self-loop vẫn bị ràng buộc phát xạ/duration từ ít mẫu. Thời gian chỉ một lần đo sau warm-up, không phải benchmark thống kê. Không thể suy phương pháp tốt hơn trên dữ liệu giọng thật từ kết quả TTS.

### E5 và lỗi robustness

Gaussian noise SNR 20 dB: HMM còn 5/10, giảm 50 điểm phần trăm. SNR tính trên toàn bộ WAV trước trim (bao gồm silence), endpoint có thể giữ thêm vùng nhiễu, nên đây là tác động kết hợp của front-end và acoustic mismatch. Các file sai:

- dong_08.wav: dong → xuong
- dung_08.wav: dung → xuong
- len_08.wav: len → xuong
- mo_07.wav: mo → xuong
- mo_08.wav: mo → xuong

Gain 0.3 vẫn 10/10: peak normalization và CMVN giảm ảnh hưởng biên độ. Không mở rộng kết luận sang clipping, reverberation hay microphone khác. Noise/gain là environment/channel mismatch; chưa có speaker mismatch.

### E6 và language model

Các λ 0, 0.1, 0.3, 1.0 không đổi dự đoán ở WAV thật với các context giả định đã thử. Có 60 trial context/λ nhưng chỉ 10 WAV, không phải 60 mẫu độc lập. Không có context ghi âm nên không tính accuracy LM. Corpus 12 chuỗi lệnh tự xây dựng; smoothing k=0.5, xác suất tổng mỗi context bằng 1.

Ví dụ **điểm nhân tạo tách biệt**: với context `mo`, top-1 acoustic `mo` đổi thành `len` ở λ≥0.1. Nếu từ đúng giả định là `len` thì LM sửa được; nếu từ đúng là `mo` thì LM làm sai. Đây là hai cách giải thích ví dụ giả lập, không phải kết quả thu âm. Kết quả thật nằm trong E6_LM_scores.csv; ví dụ nằm trong E6_artificial_illustration.csv.

### Giới hạn

Lexicon WW_* chỉ minh họa whole-word units, chưa có phone inventory chuẩn. Chưa có người nói mới, dữ liệu người thật, context liên tục hoặc OOV. Threshold reject là heuristic train in-sample, chưa xác minh unknown. Mỗi test thay đổi accuracy 10 điểm phần trăm; cần nhiều mẫu và validation độc lập để khẳng định tổng quát hóa.
