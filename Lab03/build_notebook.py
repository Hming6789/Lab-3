"""Tạo notebook với code theo từng giai đoạn; chạy bằng execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# Lab 3 Nhận dạng năm lệnh tiếng Việt bằng Gaussian HMM

CSE457 · Tài liệu gốc: `Lab 3.pdf`, trang 6–10. Chạy tuần tự trong VS Code với kernel Conda đã có.

**Dữ liệu hoàn toàn tổng hợp bằng eSpeak Vietnamese**, không phải bản ghi người thật.
Accuracy ở đây chỉ áp dụng cho 10 WAV TTS cùng nguồn. Không dùng test để fit hoặc chọn cấu hình.
Cấu hình chính được chọn trước theo PDF: 5 states, 39D, diagonal covariance.

Mã tái sử dụng nằm trong `lab03.py`; notebook thực thi từng giai đoạn và hiển thị kết quả thật.
Các CSV/PNG/JSON được ghi trong `results/`; model GaussianHMM thật được lưu bằng joblib.''')
md('''## 0 Kiểm tra môi trường

Trước khi cài thêm bất cứ gói nào, chạy trong terminal Conda:
```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```
Lần chạy bàn giao dùng thư viện đã có trong Conda, không cài/nâng cấp base.
Jupyter metapackage không có metadata, nhưng ipykernel và jupyter_client hoạt động.
VS Code chạy notebook thông qua ipykernel. Không cần Colab hoặc venv.''')
code('''from pathlib import Path
import sys
import time
from threadpoolctl import threadpool_limits
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if not (ROOT / 'lab03.py').exists():
    ROOT = ROOT / 'Lab03_HMM'
assert (ROOT / 'lab03.py').exists(), 'Mở notebook từ thư mục dự án'
sys.path.insert(0, str(ROOT))
import lab03 as lab
import pandas as pd
import numpy as np
import joblib
thread_limit = threadpool_limits(limits=1)
environment = lab.setup()
print(sys.version)
print(sys.executable)
display(pd.DataFrame(environment['packages'].items(), columns=['Package', 'Version']))''')
md(r'''## A Chuẩn hóa và trích đặc trưng

Kiểm tra file, header WAV, hash byte và hash PCM, metadata, clipping và train/test overlap trước khi fit.
Không sửa dataset gốc. `repair_formats()` chỉ tạo bản sao ở thư mục khác nếu có sai định dạng.
Endpoint dùng ngưỡng năng lượng tương đối 30 dB; vùng giữ lại được tô trên waveform.
Pre-emphasis: $y[n]=x[n]-0.97x[n-1]$.

Mỗi frame có 13 MFCC (bao gồm c0), 13 delta và 13 delta², thành 39 chiều.
MFCC mô tả phổ; delta là tốc độ thay đổi; delta² là thay đổi của tốc độ.
25 ms window Hamming, hop 10 ms, FFT 512, 26 mel bands. Librosa center=True có padding ở biên.
CMVN từng utterance: $(X-\mu_X)/\max(\sigma_X,10^{-8})$, không dùng thống kê tổng hợp test.''')
code('''audit, audit_summary = lab.audit_data()
display(audit)
print(audit_summary)
cache, shapes = lab.feature_stage(audit)
display(shapes)
for label in ['mo', 'xuong']:
    display(Image(filename=str(ROOT / 'results/feature_plots' / f'{label}.png')))''')
md(r'''## B Khởi tạo Gaussian HMM và C Baum Welch

$\lambda=(A,B,\pi)$: A là chuyển trạng thái, B là Gaussian phát xạ, π là phân bố bắt đầu.
π cố định [1,0,0,0,0]. A bắt đầu với self-loop 0.7, next-state 0.3, state cuối 1.
Chỉ cập nhật `t,m,c`; không khởi tạo lại π hoặc A. Gaussian được khởi tạo bằng chia đều thời gian của **train**.
Mỗi lớp fit `np.vstack(sequences)` với `lengths` của đúng 6 utterance.

Baum–Welch lặp Forward–Backward (E) và cập nhật kỳ vọng (M); chỉ đạt cực trị cục bộ.
Covariance có prior nhỏ để tránh suy biến. Sau fit kiểm tra π, tổng hàng, mọi chuyển cấm bằng 0.
Nếu topology không giữ được, fit lại với A cố định và ghi rõ fallback.
`monitor_.converged` có thể True khi hết 80 iteration; bảng ghi thêm `tolerance_met` và `stop_reason`.''')
code('''models, training, training_elapsed = lab.fit_models(audit, cache, persist=True)
display(training)
for label, model in models.items():
    lab.assert_topology(model)
    print(label, 'π =', model.startprob_)
    print(np.round(model.transmat_, 5))
display(Image(filename=str(ROOT / 'results/convergence/baseline.png')))''')
md('''## D Nhận dạng và Viterbi

Forward cộng xác suất của **mọi** đường hợp lệ; `model.score(X)` trả log likelihood.
Viterbi dùng max và backpointer để lấy **một** đường tốt nhất; log probability của đường không lớn hơn Forward.
Cùng utterance có cùng T cho cả 5 model, nên chia T không đổi argmax.
Gap top-1/top-2 phản ánh mức phân biệt của bộ model, không phải xác suất đúng được hiệu chuẩn.
`recognize(wav_path, models)` trả từ tiếng Việt, scores và xếp hạng.
Hai state path bên dưới dùng model nhãn thật; state là đoạn âm học ẩn, không mặc định tương ứng phone.''')
code('''predictions = lab.evaluate(audit, models, cache)
predictions.to_csv(ROOT/'results/predictions.csv', index=False, encoding='utf-8-sig')
lab.table(predictions.head(3), 'likelihood_examples')
display(predictions)
print('Clean accuracy:', predictions.correct.mean())
print(lab.recognize(ROOT/'dataset/test/mo/mo_07.wav', models))
occupancy = lab.viterbi_stage(audit, cache, models)
display(occupancy)
for label in ['mo', 'xuong']:
    display(Image(filename=str(ROOT/'results/viterbi_plots'/f'{label}.png')))
display(Image(filename=str(ROOT/'results/confusion_matrices/HMM_clean.png')))''')
md('''## E So sánh DTW và H Đánh giá robustness qua E1 đến E5

E1 thay số state 3/5/7; E2 thay số chiều 13/39; E3 thay diag/full, các lựa chọn khác giữ nguyên.
Full covariance là stress test có regularization; nhiều frame tương quan không tương đương nhiều utterance độc lập.
So sánh train/test score chỉ trong cùng chiều và cùng phép chuẩn hóa; không suy overfit từ score cao đơn thuần.

E4 dùng DTW13 kiểu Lab 2 và DTW39 để so sánh công bằng với HMM39 khi chỉ đổi recognizer.
DTW tìm template gần nhất trong 30 mẫu, Euclid và chi phí chia chiều dài đường.
Không có mã Lab 2 được cung cấp, nên đây là baseline tái xây dựng.
Thời gian dự đoán gồm đọc file, trích đặc trưng và scoring, một thread sau warm-up.
Thời gian chuẩn bị DTW chỉ lưu template; thời gian trích đặc trưng train dùng chung chưa tính vào hai bên.

E5 thêm Gaussian noise SNR 20 dB trước trim và giảm gain còn 0.3, không fit lại model.
Time-stretch 0.8/1.2 kiểm tra thay đổi tốc độ nhân tạo, không phải người nói mới.
Các điều kiện noise/gain là environment/channel mismatch. Chưa có speaker mismatch.
Không dùng kết quả test E1–E3 để thay cấu hình chính.''')
code('''configurations, comparison, robustness = lab.experiment_stage(
    audit, cache, models, training, predictions, training_elapsed)
for name in ['E1', 'E2', 'E3', 'E4', 'E5', 'speed_robustness', 'E4_paired_cases']:
    print(name)
    display(pd.read_csv(ROOT/'results/experiment_tables'/f'{name}.csv'))
print('Các WAV dự đoán sai:')
display(predictions.loc[~predictions.correct])
display(pd.read_json(ROOT/'results/HMM_clean_classification_report.json'))
lab.score_plot(predictions)
display(Image(filename=str(ROOT/'results/score_distribution.png')))
display(Image(filename=str(ROOT/'results/confusion_matrices/E2_13.png')))
display(Image(filename=str(ROOT/'results/confusion_matrices/DTW_shared39.png')))''')
md('''## F Lexicon mini

Chưa có phone inventory của giảng viên. Lexicon minh họa dùng token WW_* đại diện toàn từ giả lập,
**không phải phiên âm chuẩn và không phải phone inventory**. Parser vẫn đọc đúng định dạng WORD UNIT ...
Whole-word HMM ánh xạ thẳng nhãn → model nên không cần lexicon để chấm 5 từ đã biết.
Với phone HMM, lexicon cung cấp chuỗi phone; nối các model theo thứ tự phát âm,
chuyển từ state cuối phone trước sang state đầu phone sau, điều chỉnh xác suất exit/self-loop,
có thể thêm silence/biến thể phát âm. Không nối các absorbing state nguyên trạng vì không thể thoát phone.
Chưa huấn luyện phone HMM.''')
md(r'''## G Bigram MAP và E6 Language Model

Corpus lệnh tự xây dựng, tokenize theo khoảng trắng, ánh xạ lệnh tiếng Việt sang nhãn nội bộ.
Đếm cả `<s>` và `</s>`; tập đích V gồm 5 lệnh và `</s>`.
Mẫu số là số outgoing bigram từ context, đảm bảo tổng xác suất bằng 1.
$P(w|h)=[C(h,w)+k]/[C_{out}(h)+kV]$, k=0.5.
$S(w)=\log P(X|\lambda_w)/T + \lambda_{LM}\log P(w|h)$, λ = 0, 0.1, 0.3, 1.

Không có chuỗi lệnh đã thu, nên context h được giả định và **không báo accuracy LM như test có context thật**.
Định nghĩa acoustic gần nhau trước khi xem kết quả là gap/frame <0.5.
Ví dụ điểm nhân tạo được lưu riêng với nhãn ARTIFICIAL_SCORES_NOT_WAV để minh họa quyết định đổi do LM.
LM chỉ hỗ trợ nếu context đáng tin và corpus phù hợp; λ quá lớn có thể làm sai acoustic đang đúng.''')
code('''lexicon, lm_summary = lab.language_stage(predictions)
print('Parser:', lexicon)
display(pd.read_csv(ROOT/'language_model/unigram_counts.csv'))
display(pd.read_csv(ROOT/'language_model/bigram_counts.csv').head(18))
display(lm_summary)
display(pd.read_csv(ROOT/'results/experiment_tables/E6_LM_scores.csv').head(20))
display(pd.read_csv(ROOT/'results/experiment_tables/E6_changed_decisions.csv'))
display(pd.read_csv(ROOT/'results/experiment_tables/E6_artificial_illustration.csv'))
print((ROOT/'language_model/notes.json').read_text(encoding='utf-8'))''')
md('''## H Đề xuất reject và kiểm tra lưu mô hình

Argmax luôn chọn một trong 5 lớp dù đầu vào ngoài từ vựng.
Đề xuất reject khi top1/frame thấp hoặc gap/frame nhỏ. Ngưỡng phân vị train ở đây chỉ là heuristic in-sample,
không phải ngưỡng đã xác minh. Cần validation độc lập, từ OOV và silence/noise để đo false reject/false accept.
CMVN và chuẩn hóa peak có thể giúp bất biến gain; không chứng minh chịu được mọi channel.
10 test ⇒ mỗi file làm accuracy đổi 10 điểm phần trăm, kết luận cần thận trọng.''')
code('''reject_proposal = lab.reject_proposal(audit, cache, models)
print(reject_proposal)
for label in lab.LABELS:
    loaded = joblib.load(ROOT/'models'/f'hmm_{label}.joblib')
    lab.assert_topology(loaded)
    test_f = cache[f'test/{label}/{label}_07.wav']
    assert np.isclose(loaded.score(test_f), models[label].score(test_f))
print('5 mô hình load lại và cho cùng score; topology hợp lệ.')
lab.save_json(dict(status='completed', clean_accuracy=float(predictions.correct.mean()),
    clean_correct=int(predictions.correct.sum()), test_files=10,
    final_configuration='Preset PDF: 5 states, 39D, diag; NOT chosen on test',
    dataset='eSpeak Vietnamese TTS only', speaker_mismatch='Chưa thực hiện',
    lexicon='Illustrative WW units; instructor phone inventory unavailable',
    lm_context='Hypothetical only; no recorded previous word',
    failed_experiments=configurations.loc[configurations.status.ne('completed'),'config'].tolist()),
    ROOT/'results/run_summary.json')''')
theory=(ROOT/'reports/Theory_6_1.md').read_text(encoding='utf-8')
md(theory)
md('''## Kết luận và cách đọc sản phẩm

Xem `results/predictions.csv` cho từng file, `results/experiment_tables/E1.csv`–`E6.csv` cho bảng thí nghiệm,
`results/environment.json` cho phiên bản, `results/data_audit.csv` cho kiểm tra dữ liệu.
`reports/Bao_cao_Lab03_HMM.pdf` được tạo sau khi chạy notebook, bằng `build_report.py` trong runtime tài liệu.
Không tự thay đổi thư viện khi chỉ mở notebook. Chỉ load joblib từ nguồn tin cậy.

Nguồn: PDF giảng viên CSE457, trang 2–10; [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html).
Hạn chế bàn giao: lexicon chờ phone inventory, LM context là minh họa, chưa có giọng người thật/OOV.''')
notebook=nbf.v4.new_notebook(cells=cells, metadata=dict(
    kernelspec=dict(display_name='Python (Conda hiện có)',language='python',name='python3'),
    language_info=dict(name='python',version='3.13.12')))
nbf.write(notebook, ROOT/'Lab03.ipynb')
print('Created', ROOT/'Lab03.ipynb')
