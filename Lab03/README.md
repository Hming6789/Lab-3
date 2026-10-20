# Lab 3 Nhận dạng năm lệnh tiếng Việt bằng Gaussian HMM

Đã thực thi pipeline và **toàn bộ notebook** trong Conda có sẵn, theo `Lab 3.pdf` CSE457 (bản ở Downloads, 11 trang). Dữ liệu là **40 WAV tổng hợp bằng eSpeak Vietnamese**, không phải giọng người thật. Bản gốc ở thư mục cha được giữ nguyên; dự án có bản sao dataset để chạy độc lập.

## Mở và chạy trong VS Code trên Windows

Mở thư mục `Lab03_HMM`, mở `Lab03.ipynb`, chọn **Select Kernel → Python Environments → Conda base Python 3.13.12** đang có trên máy:

```text
C:\Users\vuhoa\miniconda3\python.exe
```

Chọn **Run All**. Notebook bàn giao đã có output bảng, waveform, heatmap, confusion matrix, Viterbi và log chạy thật. Mã các hàm được tách trong `lab03.py`; notebook gọi từng giai đoạn có giải thích A–H, E1–E6 và trả lời 10 câu mục 6.1.

Trước khi cân nhắc cài thư viện, kiểm tra đúng terminal Conda:

```powershell
conda activate base
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
python -c "import numpy, scipy, matplotlib, librosa, soundfile, sklearn, hmmlearn, pandas, joblib, ipykernel, jupyter_client; print('Imports OK')"
```

Lần bàn giao **không cài, gỡ hoặc nâng cấp gì trong Conda base**. Các thư viện đã tồn tại và chạy được. Metapackage `jupyter` chưa có metadata nhưng `ipykernel`/`jupyter_client` đủ chạy notebook trong VS Code và đã kiểm chứng. `requirements.txt` ghi các phiên bản đang dùng cùng metapackage Jupyter để dựng môi trường đầy đủ khi cần. Không chạy cài cả requirements vào base một cách tự động. Nếu thiếu gói trong môi trường Conda dự án được người dùng chọn, chỉ cài gói thiếu bằng `python -m pip install <package>` sau khi kiểm tra interpreter.

Nếu `python` vẫn trỏ WindowsApps, dùng đường dẫn đầy đủ trong PowerShell, từ thư mục dự án:

```powershell
& 'C:\Users\vuhoa\miniconda3\python.exe' .\lab03.py
```

Lệnh trên chạy toàn bộ audit, feature, huấn luyện, E1–E6 và xuất kết quả, **không cần notebook server**. Mỗi lần chạy sẽ ghi đè kết quả sinh ra, không sửa WAV. Model chính luôn là 5 states, 39D, diag theo PDF; không chọn lại bằng test.

## Kết quả đã đo

| Điều kiện / phương pháp | Đúng / test | Accuracy |
|---|---:|---:|
| HMM 5 states, 39D, diag, sạch | 10/10 | 100% |
| DTW13 / DTW39, sạch | 10/10 | 100% |
| HMM 3 hoặc 7 states, sạch | 10/10 | 100% |
| HMM MFCC13, sạch | 10/10 | 100% |
| HMM full covariance regularized, sạch | 10/10 | 100% |
| HMM nhiễu Gaussian SNR 20 dB | 5/10 | 50% |
| HMM gain 0.3 | 10/10 | 100% |
| HMM time-stretch rate 0.8 / 1.2 | 8/10 và 9/10 | 80% / 90% |
| DTW39 time-stretch rate 0.8 / 1.2 | 10/10 | 100% / 100% |

Thời gian thực tế nằm trong `results/experiment_tables/E4.csv`; đó là một lần đo, một thread, sau warm-up, gồm đọc file + feature + nhận dạng. Chuẩn bị HMM là tổng fit; chuẩn bị DTW là tạo template list, cả hai chưa tính feature train dùng chung. DTW13 là baseline MFCC tái xây dựng vì không có mã Lab 2; DTW39 cùng front-end 39D cho so sánh chỉ đổi recognizer.

E1 chỉ đổi states; E2 chỉ đổi số chiều; E3 chỉ đổi covariance cùng quy tắc khởi tạo/prior; full là stress test có regularization vì ít dữ liệu. Full có train score/frame 52.77 nhưng test nhãn thật −561.79, trong khi diag −16.95/−29.72: dấu hiệu khớp train quá mạnh và thiếu ổn định, dù accuracy argmax vẫn 100%. Không so trực tiếp score 13D với 39D.

Ba model chính `mo`, `dong`, `len` có log likelihood giảm nhẹ ở vòng cuối; hai model còn lại đạt tolerance. Cảnh báo được ghi đầy đủ, không gọi cả năm model hội tụ chỉ vì `monitor.converged=True`. Topology của cả năm được kiểm tra và giữ đúng; không cần fallback cố định A. Tất cả model đều có score hữu hạn.

E6 đã thử λ = 0, 0.1, 0.3, 1 với 6 context giả định/utterance. Không có quyết định WAV đổi và không file nào có gap/frame <0.5. Ví dụ riêng với **điểm acoustic nhân tạo** cho thấy LM đổi từ `mo` sang `len`; không giả làm WAV. Chưa có context câu ghi âm nên không báo accuracy LM như test có context thật.

## Sản phẩm

```text
Lab03_HMM/
  Lab03.ipynb                  # Notebook đã chạy, có output
  lab03.py                     # Pipeline và recognize()
  README.md
  requirements.txt
  dataset/                     # 30 train + 10 test, metadata + manifest
  models/                      # 5 GaussianHMM .joblib + metadata JSON
  results/
    data_audit.csv             # Header, duration, hash, clipping, metadata
    data_audit_summary.json
    environment.json
    run_summary.json
    verification.json
    predictions.csv            # Mỗi test: 5 scores, top1/top2, gap
    feature_plots/             # Waveform, MFCC, delta, delta² cho 2 từ
    viterbi_plots/             # 2 path PNG + chuỗi state CSV
    confusion_matrices/        # Clean, từng cấu hình, noise/gain/speed
    convergence/               # Log likelihood theo iteration
    experiment_tables/         # E1.csv ... E6.csv và chẩn đoán chi tiết
  language_model/
    lexicon.txt
    lexicon_parsed.json
    corpus.txt
    unigram_counts.csv
    bigram_counts.csv
    notes.json
  reports/
    Bao_cao_Lab03_HMM.pdf       # Báo cáo 6 trang gồm 10 câu hỏi
    Theory_6_1.md              # Trả lời chi tiết mục 6.1
    Analysis_results.md        # Phân tích đo được, lỗi robustness
```

`models/hmm_*.joblib` chứa GaussianHMM thật, không phải placeholder. Chỉ load joblib từ nguồn tin cậy. Model JSON ghi danh sách file train, lengths, π và A; `results/environment.json` ghi toàn bộ phiên bản.

```python
from pathlib import Path
import joblib
from lab03 import recognize, LABELS
models = {label: joblib.load(Path('models') / f'hmm_{label}.joblib') for label in LABELS}
word, scores, ranking = recognize(Path('dataset/test/mo/mo_07.wav'), models)
print(word, scores, ranking)
```

## Kiểm chứng và tái tạo sản phẩm

```powershell
# Kiểm tra model, ranh giới train/test, output notebook và bảng kết quả
& 'C:\Users\vuhoa\miniconda3\python.exe' .\verify_submission.py

# Thực thi notebook bằng cùng Python, không cần nbclient/nbconvert
& 'C:\Users\vuhoa\miniconda3\python.exe' .\execute_notebook.py
```

Runner notebook cần kết nối loopback cục bộ của Jupyter. Cache Numba/Matplotlib nằm trong dự án, không trong site-packages Conda. Nếu môi trường sandbox chặn socket, chạy runner trong terminal VS Code bình thường. Không cần sửa thư viện. Log `notebook_execution.log` xác nhận từng ô.

`build_notebook.py` tạo lại notebook từ source và **xóa output cũ**, chỉ chạy khi muốn dựng lại cấu trúc. Sau đó cần chạy `execute_notebook.py`.

Báo cáo PDF được dựng từ kết quả CSV/JSON bằng `build_report.py`, sử dụng runtime tài liệu bundled của Codex đã có reportlab/PDFium, không cài thêm vào Conda:

```powershell
& 'C:\Users\vuhoa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\build_report.py
```

PDF đã được raster hóa và xem từng trang. Các PNG QA và cache/runtime logs là phụ trợ, không phải mô hình hay số liệu đánh giá. Không cần nộp chúng. Gói ZIP bàn giao loại bỏ cache, file kết nối kernel và ảnh QA.

## Giới hạn cần ghi rõ khi nộp

- Dữ liệu toàn bộ là eSpeak TTS. Không suy rộng accuracy 100% sang giọng người thật; 10 test là rất nhỏ, không tách theo người nói.
- Chưa có kiểm định người nghe độc lập về nhãn/phát âm. Metadata/hash chỉ xác minh cấu trúc và tính nhất quán.
- Chưa có phone inventory giảng viên: `WW_*` là đơn vị toàn từ giả lập, **không phải phone tiếng Việt chuẩn**. Lexicon chỉ minh họa parser và chờ xác nhận.
- Corpus LM tự xây dựng, context giả định. Không có decoder liên tục hoặc ranh giới từ thực.
- Noise/gain là environment/channel mismatch. **Chưa thực hiện speaker mismatch** vì không có người nói khác.
- Chưa có OOV; threshold score/frame và gap chỉ là đề xuất heuristic từ train in-sample. Chưa xác nhận hiệu quả unknown.

Nguồn: `Lab 3.pdf`, CSE457, mục 4–7; [hmmlearn tutorial chính thức](https://hmmlearn.readthedocs.io/en/stable/tutorial.html). Kết quả TTS, minh họa LM và ký hiệu lexicon được phân biệt rõ trong notebook/báo cáo.
