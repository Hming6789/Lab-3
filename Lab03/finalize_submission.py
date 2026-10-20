"""Phân tích số đo và lập gói nộp; không huấn luyện hoặc tạo số liệu giả."""
from pathlib import Path
import json
import hashlib
import csv
import nbformat
import zipfile
import os

ROOT=Path(__file__).resolve().parent

def submission_files(root):
    """Bỏ cache, Git metadata và toàn bộ checkout lồng khỏi gói nộp."""
    excluded={'.git','runtime_cache','kernel_runtime','report_qa','__pycache__','.ipynb_checkpoints'}
    files=[]
    for directory, dirs, names in os.walk(root):
        base=Path(directory)
        dirs[:]=sorted(d for d in dirs if d not in excluded and not (base/d/'.git').exists())
        files.extend(base/name for name in sorted(names) if name not in excluded)
    return sorted(files)
def rows(name):
    with (ROOT/'results/experiment_tables'/f'{name}.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def failures(name):
    return [f"{Path(r['path']).name}: {r['truth']} → {r['predicted']}" for r in rows(name) if r['truth']!=r['predicted']]
def pct(x):return f'{float(x)*100:.0f}%'
configs=rows('configuration_results')
train=rows('baseline_training')
comparison=rows('E4')
text='''## Phân tích kết quả đo được

### Dữ liệu và đặc trưng

40 WAV eSpeak, 30 train/10 test, mọi header và metadata khớp. Không trùng byte hoặc PCM giữa train/test, không có clipping theo ngưỡng kiểm tra. Chưa xác nhận bằng nghe. Hash khác nhau không chứng minh độc lập thống kê giữa các WAV cùng bộ TTS.

Hai ví dụ Viterbi `mo_07` có 25 frame, `xuong_07` có 28 frame sau endpoint detection. MFCC biểu diễn khác biệt phổ; delta/delta² nêu diễn biến quanh chuyển đoạn trên heatmap. Mỗi đặc trưng là T×39, CMVN trong từng utterance.

### HMM và hội tụ

Mô hình chính đạt 10/10 test sạch. Không có cặp nhầm lẫn trên tập sạch. Score gap/frame nhỏ nhất vẫn lớn hơn 0.5, không có WAV acoustic gần nhau theo tiêu chí đặt trước. Các likelihood là mật độ của feature liên tục, không phải xác suất dự đoán được hiệu chuẩn.

'''
for r in train:
    text+=f"- `{r['label']}`: {r['iterations']} iteration, dừng `{r['stop_reason']}`, delta cuối {float(r['last_improvement']):.6f}; A hợp lệ, π cố định.\n"
text+='''
Likelihood giảm nhẹ ở một số model: không gọi chúng hội tụ tốt chỉ từ cờ `monitor.converged`. Prior/regularization và tối ưu số học có thể làm mục tiêu likelihood monitor giảm nhẹ; mọi cảnh báo được giữ, không che đi. Cần thêm dữ liệu/validation và kiểm tra nhiều khởi tạo nếu cải tiến hệ thống.

### E1 đến E3

'''
for r in configs:
    text+=f"- `{r['config']}`: accuracy {pct(r['accuracy'])}; {float(r['mean_iterations']):.1f} vòng trung bình; {r['tolerance_models']}/5 model đạt tolerance; train/test true-label score/frame {float(r['train_score_per_frame']):.2f}/{float(r['test_true_score_per_frame']):.2f}.\n"
text+='''
E1: tăng state 3→5→7 làm train score tăng, nhưng accuracy sạch đều bằng 100%, chưa có bằng chứng cần nhiều state hơn. 7 states không model nào đạt tolerance dương không giảm, nên không chọn 7 chỉ vì train likelihood cao. Giữ 5 states theo PDF đã chốt trước.

E2: cả 13 và 39 chiều đều 100%, recall cả 5 lớp đều 1.0; chưa thấy lớp nào được delta cải thiện trên tập test sạch này. Không so trị tuyệt đối likelihood giữa hai số chiều.

E3: full covariance có chênh lệch train/test rất lớn (52.77 so với −561.79/frame), trong khi diag −16.95/−29.72. Đây là dấu hiệu phù hợp với overfit và covariance thiếu ổn định. Có regularization nên vẫn chấm hữu hạn; accuracy 100% chỉ nói thứ hạng trong 5 lớp còn đúng, không chứng minh full tổng quát hóa tốt. Số frame mỗi state nhỏ so với 780 tham số covariance; frame cũng tương quan, nên full chỉ là stress test có prior.

### E4 và tốc độ nói

'''
for r in comparison:
    text+=f"- {r['method']}: {pct(r['accuracy'])}, chuẩn bị {float(r['preparation_s']):.6f} s, dự đoán {float(r['prediction_ms']):.3f} ms/file; 6 template/train utterance mỗi lớp.\n"
text+='''
Không có file sạch mà HMM39 và DTW39 khác quyết định (E4_paired_cases.csv). Đối chiếu có kiểm soát tốc độ với DTW39 (cùng 39D):

'''
for rate in [.8,1.2]:
    errors=failures(f'HMM_rate_{rate}_predictions')
    text+=f"- Rate {rate}: HMM sai {len(errors)}/10; DTW39 đúng 10/10. Các file HMM sai: "+'; '.join(errors)+'.\n'
text+='''
DTW warping linh hoạt hơn trên các biến đổi tốc độ nhân tạo này; HMM self-loop vẫn bị ràng buộc phát xạ/duration từ ít mẫu. Thời gian chỉ một lần đo sau warm-up, không phải benchmark thống kê. Không thể suy phương pháp tốt hơn trên dữ liệu giọng thật từ kết quả TTS.

### E5 và lỗi robustness

Gaussian noise SNR 20 dB: HMM còn 5/10, giảm 50 điểm phần trăm. SNR tính trên toàn bộ WAV trước trim (bao gồm silence), endpoint có thể giữ thêm vùng nhiễu, nên đây là tác động kết hợp của front-end và acoustic mismatch. Các file sai:

'''
for item in failures('noise_20dB_predictions'):text+='- '+item+'\n'
text+='''
Gain 0.3 vẫn 10/10: peak normalization và CMVN giảm ảnh hưởng biên độ. Không mở rộng kết luận sang clipping, reverberation hay microphone khác. Noise/gain là environment/channel mismatch; chưa có speaker mismatch.

### E6 và language model

Các λ 0, 0.1, 0.3, 1.0 không đổi dự đoán ở WAV thật với các context giả định đã thử. Có 60 trial context/λ nhưng chỉ 10 WAV, không phải 60 mẫu độc lập. Không có context ghi âm nên không tính accuracy LM. Corpus 12 chuỗi lệnh tự xây dựng; smoothing k=0.5, xác suất tổng mỗi context bằng 1.

Ví dụ **điểm nhân tạo tách biệt**: với context `mo`, top-1 acoustic `mo` đổi thành `len` ở λ≥0.1. Nếu từ đúng giả định là `len` thì LM sửa được; nếu từ đúng là `mo` thì LM làm sai. Đây là hai cách giải thích ví dụ giả lập, không phải kết quả thu âm. Kết quả thật nằm trong E6_LM_scores.csv; ví dụ nằm trong E6_artificial_illustration.csv.

### Giới hạn

Lexicon WW_* chỉ minh họa whole-word units, chưa có phone inventory chuẩn. Chưa có người nói mới, dữ liệu người thật, context liên tục hoặc OOV. Threshold reject là heuristic train in-sample, chưa xác minh unknown. Mỗi test thay đổi accuracy 10 điểm phần trăm; cần nhiều mẫu và validation độc lập để khẳng định tổng quát hóa.
'''
(ROOT/'reports/Analysis_results.md').write_text(text,encoding='utf-8')
nb=nbformat.read(ROOT/'Lab03.ipynb',as_version=4)
nb.cells=[c for c in nb.cells if not (c.cell_type=='markdown' and c.source.startswith('## Phân tích kết quả đo được'))]
idx=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Trả lời đúng 10'))
nb.cells.insert(idx,nbformat.v4.new_markdown_cell(text))
nbformat.write(nb,ROOT/'Lab03.ipynb')

files=submission_files(ROOT)
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files
          if p.name!='submission_manifest.json'}
(ROOT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
archive=ROOT.parent/'Lab03_HMM_submission.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p,Path('Lab03_HMM')/p.relative_to(ROOT))
    if ROOT/'submission_manifest.json' not in files:
        z.write(ROOT/'submission_manifest.json','Lab03_HMM/submission_manifest.json')
print(f'Packaged {len(files)} files: {archive} ({archive.stat().st_size} bytes)')
