"""Tạo báo cáo PDF 6 trang từ kết quả đo; chạy trong runtime tài liệu bundled."""
from pathlib import Path
import csv
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/Bao_cao_Lab03_HMM.pdf'
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
W,H=595.28,841.89
M=43
CW=W-2*M
style=ParagraphStyle('Body',fontName='Arial',fontSize=10,leading=14,spaceAfter=7)
small=ParagraphStyle('Small',fontName='Arial',fontSize=8.1,leading=10.5)
title_style=ParagraphStyle('Title',fontName='ArialBold',fontSize=21,leading=27)
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('Lab 3 Nhận dạng năm lệnh tiếng Việt bằng Gaussian HMM')
c.setAuthor('Báo cáo thực nghiệm CSE457')
y=H-M

def read_json(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def rows(name):
    with (ROOT/'results/experiment_tables'/f'{name}.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def pct(x): return f'{100*float(x):.0f}%'
def fnum(x,n=2): return f'{float(x):.{n}f}'
def p(text,bold=False,size=None):
    global y
    st=ParagraphStyle('Local',parent=style,fontName='ArialBold' if bold else 'Arial',
                      fontSize=size or 10,leading=(size or 10)*1.4)
    para=Paragraph(text,st)
    _,h=para.wrap(CW,1000)
    if y-h < 43: raise ValueError(f'Text tràn trang: {text[:70]}')
    para.drawOn(c,M,y-h);y-=h+7
def heading(text):
    global y
    y-=4
    p(text,True,12)
def tbl(headers,data,widths=None):
    global y
    content=[[Paragraph(escape(str(x)),small) for x in headers]]
    content += [[Paragraph(escape(str(x)),small) for x in row] for row in data]
    t=Table(content,colWidths=widths or [CW/len(headers)]*len(headers),hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf3')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9db1bf')),
        ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe2e8'))]))
    _,h=t.wrap(CW,1000)
    if y-h <43: raise ValueError('Bảng tràn trang')
    t.drawOn(c,M,y-h);y-=h+10
def img(path,width=CW,height=None,x=M):
    global y
    im=ImageReader(str(ROOT/path));iw,ih=im.getSize()
    height=height or width*ih/iw
    if y-height<43:raise ValueError(f'Ảnh tràn trang {path}')
    c.drawImage(im,x,y-height,width=width,height=height,preserveAspectRatio=True,anchor='c')
    y-=height+7
def footer(num):
    c.setFont('Arial',8);c.setFillColor(colors.HexColor('#536472'))
    c.drawString(M,25,'CSE457  |  Lab 3  |  Dữ liệu eSpeak TTS')
    c.drawRightString(W-M,25,f'{num} / 6');c.setFillColor(colors.black)
def newpage(num,title):
    global y
    footer(num-1);c.showPage();y=H-M
    p(title,True,17)

summary=read_json('results/run_summary.json')
audit=read_json('results/data_audit_summary.json')
environment=read_json('results/environment.json')
config=rows('configuration_results')
train=rows('baseline_training')
pred=rows('HMM_clean_predictions')
compare=rows('E4')
robust=rows('E5')
viterbi=rows('viterbi_occupancy')
lm=rows('E6')
notes=read_json('language_model/notes.json')

p('Lab 3 Nhận dạng tiếng nói bằng Gaussian HMM',True,21)
p('Năm lệnh tiếng Việt  •  CSE457  •  Thực nghiệm ngày 20 tháng 10 năm 2026',size=10)
heading('1 Bài toán và phạm vi kết quả')
p(f'Hệ thống phân loại các lệnh mở, đóng, lên, xuống và dừng bằng một HMM cho mỗi từ. '
  f'Cấu hình chính đạt {summary["clean_correct"]}/10 dự đoán đúng ({pct(summary["clean_accuracy"])}). '
  'Đây là kết quả trên các WAV tổng hợp bằng eSpeak Vietnamese; không đại diện cho khả năng nhận dạng giọng người thật.')
heading('2 Dữ liệu và kiểm định')
tbl(['Nhãn','Từ','Train','Test'],[[a,b,6,2] for a,b in [('mo','mở'),('dong','đóng'),('len','lên'),('xuong','xuống'),('dung','dừng')]])
p(f'Đã đọc đủ {audit["files"]} WAV, so khớp đường dẫn, nhãn, thời lượng và SHA256 với metadata. '
  f'Tất cả mono, 16 kHz, PCM 16-bit; thời lượng {audit["duration_min_s"]:.3f}–{audit["duration_max_s"]:.3f} giây. '
  'Không có file rỗng/lỗi đọc, trùng byte hoặc PCM giữa các tập, hoặc mẫu clipping theo ngưỡng kiểm tra. '
  'Không sửa bộ dữ liệu gốc. Nhãn được kiểm tra bằng thư mục và metadata, chưa có kiểm định nghe độc lập.')
p('Manifest nói mẫu độc lập nhưng mọi file cùng nguồn TTS và có thể chia sẻ đặc tính bộ tổng hợp. '
  'Hash khác nhau không chứng minh độc lập thống kê. Khoảng lặng đầu/cuối được đo gần đúng và lưu trong data_audit.csv; '
  'không khẳng định mọi file đáp ứng 0,2–0,5 giây im lặng mà PDF đề xuất.')
heading('3 Pipeline và cách tái lập')
p('WAV → mono/resample khi cần → peak 0,95 → trim 30 dB → pre-emphasis 0,97 → MFCC + delta + delta² → CMVN từng utterance → HMM. '
  'MFCC: 13 hệ số gồm c0, FFT 512, cửa sổ Hamming 400 mẫu (25 ms), hop 160 mẫu (10 ms), 26 mel bands, 20–8000 Hz. '
  'Delta dùng hồi quy tối đa 9 frame. Feature có dạng T × 39; mỗi file có T riêng.')
p('CMVN dùng trung bình và độ lệch chuẩn của chính utterance, không gộp test vào thống kê train. '
  'Mô hình chính được chốt trước theo PDF: 5 trạng thái, 39 chiều, diag. Các E1–E3 chỉ là đánh giá so sánh trên test, không chọn lại cấu hình.')
p(f'Python Conda {environment["python"].split()[0]}, hmmlearn {environment["packages"]["hmmlearn"]}, '
  f'librosa {environment["packages"]["librosa"]}, seed 457. Không cài/nâng cấp Conda base. '
  'Chạy Lab03.ipynb từ đầu đến cuối hoặc python lab03.py; README mô tả kernel và thư viện.',size=9)

newpage(2,'4 Kiến trúc và huấn luyện HMM')
p('λ = (A, B, π). A mô tả chuyển state, B là Gaussian phát xạ, π là phân bố bắt đầu. '
  'Giả thiết Markov bậc một và độc lập quan sát có điều kiện theo state. π = [1,0,0,0,0] cố định. '
  'A chỉ có self-loop/next-state, khởi tạo 0,7/0,3; state cuối absorbing. Mean/covariance khởi tạo từ các đoạn thời gian đều của train.')
p('Mỗi model dùng đúng 6 file cùng lớp; nối bằng vstack và truyền lengths để giữ ranh giới utterance. '
  'Baum–Welch cập nhật transition, mean, covariance; tối đa 80 vòng, tol 0,001, log-domain. '
  'Prior covariance 0,01 và min_covar khởi tạo 0,001 giúp ổn định số học. Không dùng test trong fit.')
tbl(['Model','Frames','Vòng','Dừng','Train score/frame'],
    [[r['label'],r['frames'],r['iterations'],r['stop_reason'],fnum(r['train_score_per_frame'])] for r in train],
    [65,65,50,130,CW-310])
p('Đã kiểm tra π, tổng từng hàng A = 1, mọi chuyển cấm vẫn bằng 0 và state cuối tự lặp bằng 1. '
  f'Số model cần fallback cố định A: {sum(r["transition_fixed_fallback"]=="True" for r in train)}. '
  'A và π đầy đủ nằm trong models/hmm_*.json và trong output notebook. '
  'monitor.converged có thể True khi chạm giới hạn; báo cáo dùng stop_reason và delta thực để phân biệt.')
img('results/convergence/baseline.png',width=CW,height=235)
p('Đường log likelihood train theo iteration của năm model. Dấu hiệu hội tụ là cải thiện nhỏ không âm; '
  'history giảm hoặc chạm giới hạn được giữ trong all_training_diagnostics.csv, không xóa cảnh báo.',size=9)
heading('Ý nghĩa đặc trưng động')
p('13 MFCC mô tả dạng phổ, 13 delta mô tả hướng thay đổi, 13 delta² mô tả biến đổi của hướng đó. '
  'HMM mô hình hóa state theo thời gian nhưng emission chuẩn không trực tiếp nhìn frame lân cận; '
  'đặc trưng động đưa thông tin đó vào vector phát xạ. Heatmap và waveform hai từ được lưu ở feature_plots và notebook.')

newpage(3,'5 Nhận dạng likelihood và Viterbi')
p('Quyết định ML chọn model có log P(X|λ) lớn nhất. Forward cộng mọi đường, Viterbi chọn một đường. '
  'Cùng X có T cố định nên chia score cho T không đổi top-1; gap không phải xác suất tin cậy đã hiệu chuẩn.')
tbl(['File','Thật','Top 1','Top 2','Gap/frame'],
    [[Path(r['path']).name,r['truth'],r['predicted'],r['top2'],fnum(r['gap_per_frame'])] for r in pred[:3]],
    [120,60,60,60,CW-300])
p('Bảng đầy đủ năm likelihood cho mọi WAV nằm trong predictions.csv và likelihood_examples.csv. '
  'Classification report cùng confusion matrix được xuất tự động. Mỗi lớp có 2 test, nên một lỗi làm recall lớp giảm 50 điểm phần trăm.',size=9)
img('results/confusion_matrices/HMM_clean.png',width=285,height=237,x=(W-285)/2)
img('results/viterbi_plots/mo.png',width=CW,height=142)
img('results/viterbi_plots/xuong.png',width=CW,height=142)
vp=[]
for label in ['mo','xuong']:
    vv=[r for r in viterbi if r['model']==label]
    vp.append(label+': '+', '.join(r['frames'] for r in vv)+' frame/state')
p('; '.join(vp)+'. Một frame bước 10 ms; state path dùng model nhãn thật. '
  'Không diễn giải các state thành phoneme. Cả hai path tăng đơn điệu và Viterbi logp ≤ Forward logp.',size=8.5)

newpage(4,'6 Thí nghiệm có kiểm soát E1 đến E5')
tbl(['TN','Cấu hình','Test acc','Train acc','Vòng TB','Train / test score'],
    [[('E1' if r['config'] in ['E1_3','E1_7'] else 'E2' if r['config']=='E2_13' else 'E3' if r['config']=='E3_full' else 'Gốc'),
      f'{r["states"]} state / {r["dimensions"]}D / {r["covariance"]}',pct(r['accuracy']),pct(r['train_accuracy']),
      fnum(r['mean_iterations'],1),fnum(r['train_score_per_frame'])+' / '+fnum(r['test_true_score_per_frame'])] for r in config],
    [30,145,53,53,55,CW-336])
p('E1 chỉ đổi số state. Số state lớn có thể tăng khả năng phân chia thời gian nhưng tăng tham số và nguy cơ nghiệm cục bộ. '
  'E2 chỉ đổi 13/39 chiều. Không so trị tuyệt đối likelihood giữa hai số chiều để kết luận tốt/xấu. '
  'E3 đổi diag/full với cùng 39 chiều; full dùng prior ma trận để ổn định. '
  'Full có 780 covariance parameters/state so với 39 diag; chỉ 6 utterance/lớp nên đây là stress test có regularization, không chứng minh đủ dữ liệu độc lập.')
p('Cả 3/5/7 state và 13/39 chiều đều đạt 100% test sạch, nên chưa thấy lớp nào được delta cải thiện về accuracy. '
  'Số model đạt tolerance lần lượt 1/5, 2/5, 0/5 ở 3/5/7 state. '
  'Full có train score/frame 52,77 nhưng test nhãn thật −561,79, so với diag −16,95/−29,72; '
  'khoảng cách lớn cho thấy full khớp train rất mạnh và thiếu ổn định ngoài train dù argmax test vẫn đúng.',size=9)
tbl(['E4 Method','Chiều','Accuracy','Chuẩn bị s','Dự đoán ms/file'],
    [[r['method'],r['dimensions'],pct(r['accuracy']),fnum(r['preparation_s'],4),fnum(r['prediction_ms'])] for r in compare],
    [95,48,70,110,CW-323])
p('DTW13 tái xây dựng baseline Lab 2; DTW39 giữ cùng front-end với HMM39 để chỉ đổi bộ nhận dạng. '
  'DTW chọn mẫu gần nhất trong 30 mẫu bằng Euclidean distance, chuẩn hóa chi phí theo độ dài đường. '
  'Thời gian gồm đọc WAV, feature và scoring; một lần đo sau warm-up, một thread, phụ thuộc máy. '
  'Chuẩn bị không gồm trích feature train dùng chung; HMM là tổng thời gian fit, DTW là tạo danh sách template.')
paired=rows('E4_paired_cases')
diff=[r for r in paired if r['predicted_HMM']!=r['predicted_DTW']]
p(('Các file HMM/DTW13 khác quyết định: '+', '.join(Path(r['path']).name for r in diff)+'.') if diff else
  'Không có file HMM và DTW13 khác quyết định trong 10 test sạch; không tạo ra trường hợp một phương pháp thắng phương pháp kia.')
tbl(['E5 Điều kiện','Accuracy','Giảm điểm %'],[[r['condition'],pct(r['accuracy']),fnum(r['drop_pp'])] for r in robust])
speed=rows('speed_robustness')
p('Thay đổi tốc độ nhân tạo bằng time-stretch: '+ '; '.join(f'{r["method"]}, rate {r["rate"]}: {pct(r["accuracy"])}' for r in speed)+'.',size=9)
p('Nhiễu Gaussian SNR 20 dB được thêm trước trim; gain còn 0,3. Đây là mismatch môi trường/kênh, không phải speaker mismatch. '
  'Chuẩn hóa peak và CMVN làm giảm tác động gain. Chưa có người nói mới. '
  'Không thể kết luận phương pháp tốt hơn trên giọng thật từ dữ liệu TTS; DTW hợp bài toán rất ít mẫu/từ vựng nhỏ, '
  'HMM có lợi về mô hình xác suất và khả năng ghép subword khi có đủ dữ liệu.',size=9)

newpage(5,'7 Lexicon Bigram và E6')
p('Chưa có phone inventory từ giảng viên. lexicon.txt dùng “mở WW_MO”, “đóng WW_DONG”, “lên WW_LEN”, '
  '“xuống WW_XUONG”, “dừng WW_DUNG” để minh họa parser WORD UNIT…; các WW_* là đơn vị toàn từ giả lập, '
  'không phải phiên âm chuẩn. Parser đã đọc được đủ 5 mục. Cần thay bằng inventory đã xác nhận trước khi làm phone HMM.')
p('Whole-word model đã đại diện từng nhãn nên nhận dạng trực tiếp không cần lexicon. '
  'Với phone HMM, lexicon chỉ thứ tự ghép; cần thêm chuyển exit giữa các phone, phân bổ lại self-loop/exit '
  'và có thể thêm silence. Không thể chỉ giữ state cuối absorbing của từng phone rồi nối mà vẫn cho phép thoát.')
heading('Bigram và tái xếp hạng')
p('Corpus tự xây dựng gồm 12 dòng chuỗi lệnh, token theo khoảng trắng và ánh xạ nhãn nội bộ. '
  'Có &lt;s&gt;/&lt;/s&gt;; tập đích V = 6 gồm 5 lệnh và &lt;/s&gt;. '
  'P(w|h) = [C(h,w)+0,5] / [C_out(h)+0,5V]. Đã kiểm tra tổng xác suất bằng 1 cho từng context. '
  'S(w) = log P(X|λw)/T + λLM log P(w|h).')
tbl(['λLM','Số trial context giả định','Đổi so với acoustic'],
    [[r['lm_weight'],r['trials'],r['changed_decisions']] for r in lm])
p(f'Acoustic gần nhau được định nghĩa trước là gap/frame &lt; 0,5: có {len(notes["close_test_files"])} file thỏa. '
  'Mỗi WAV được thử 6 context giả định, không phải 60 mẫu test độc lập. '
  'Không có previous-word context ghi âm, nên không báo accuracy LM như đánh giá thực tế. '
  'Bảng E6_LM_scores.csv lưu acoustic, LM, tổng score và dự đoán cho từng λ/context/candidate.')
art=rows('E6_artificial_illustration')
tbl(['Ví dụ điểm nhân tạo','λ 0','λ 0,1','λ 0,3','λ 1'],
    [['Context mở; điểm sát nhau']+[next(r['prediction'] for r in art if float(r['lm_weight'])==v) for v in [0,.1,.3,1]]],
    [200]+[(CW-200)/4]*4)
p('Ví dụ trên dùng điểm acoustic tự đặt, lưu riêng với nguồn ARTIFICIAL_SCORES_NOT_WAV, không phải kết quả WAV. '
  'LM có thể sửa lỗi khi context đúng và corpus phù hợp; λ lớn hoặc context sai có thể lấn át bằng chứng âm học. '
  'Thang acoustic/frame ở đây theo ví dụ mục 5.9 PDF, không tương đương giữ nguyên trọng số trên score toàn chuỗi.')
heading('8 Hạn chế và kết luận')
p('Bộ test chỉ 10 file (mỗi file tương ứng 10 điểm phần trăm accuracy), đều cùng bộ tổng hợp. '
  'Chưa có giọng thật, speaker mismatch, OOV hay context câu thực; lexicon chưa được giảng viên xác nhận. '
  'Argmax luôn buộc chọn một nhãn. Đề xuất reject theo score/frame thấp hoặc gap nhỏ; phân vị 5% trên train chỉ là heuristic, '
  'chưa xác minh false accept/false reject. Cần validation và OOV độc lập để hiệu chuẩn.')
p('Kết quả và cảnh báo được giữ trong CSV/JSON; notebook đã thực thi tuần tự và năm model được load lại để kiểm tra score/topology. '
  'Đây là hệ nhận dạng lệnh TTS phục vụ thực hành HMM; chưa đủ bằng chứng triển khai nhận dạng người nói thực tế.')
p('Nguồn: Lab 3 CSE457, trang 2–10; hmmlearn tutorial 0.3.3 (hmmlearn.readthedocs.io/en/stable/tutorial.html). '
  'Dữ liệu: generation_manifest.json và metadata.csv đi kèm.',size=8.5)

newpage(6,'9 Trả lời mười câu hỏi mục 6 1')
answers=[
('Forward và Viterbi','Forward dùng tổng (logsumexp trong miền log), tính P(X|λ) qua mọi đường. Viterbi dùng max và backpointer, lấy đường Q* có xác suất joint lớn nhất; hai bài toán không cho cùng score.'),
('Trách nhiệm γ và ξ','γ_t(i)=P(q_t=i|X,λ) là trách nhiệm state i cho frame t. ξ_t(i,j)=P(q_t=i,q_(t+1)=j|X,λ) là trách nhiệm chuyển i→j tại t. Chúng là số đếm mềm của E-step để cập nhật A và Gaussian trong M-step.'),
('Baum Welch hội tụ xấu','EM tìm cực trị cục bộ. Khởi tạo kém, ít dữ liệu nhưng nhiều tham số, covariance suy biến, nhãn sai hoặc quá nhiều silence đều có thể dẫn tới nghiệm kém. Kiểm tra history, thử khởi tạo trên train và chọn bằng validation, không chọn trên test.'),
('Ưu tiên covariance diag','D=39: diag có 39 phương sai/state, full có 39×40/2=780 tham số covariance/state. Full ước lượng nhiều tương quan, khó ổn định với 6 utterance và frame tương quan. Regularization giúp số học nhưng không tạo thêm dữ liệu.'),
('Delta vẫn hữu ích','HMM mô tả động học state nhưng emission giả định frame độc lập có điều kiện. Delta/delta² đưa hướng biến đổi phổ cục bộ vào từng vector. Chúng tăng chiều và tương quan nên không bảo đảm accuracy luôn tăng.'),
('Khi nào cần lexicon','Whole-word HMM có một model/nhãn và từ vựng nhỏ thì score trực tiếp không cần lexicon. Với phone/subword dùng chung, cần lexicon hoặc ánh xạ phát âm tương đương để xác định chuỗi đơn vị của mỗi từ; inventory phải thống nhất.'),
('Nói chậm và self loop','Self-loop giữ một state qua nhiều frame, kéo dài đoạn âm học. Với aᵢᵢ cố định và state không cuối, duration hình học có kỳ vọng 1/(1−aᵢᵢ) frame. State cuối absorbing bị chặn bởi kết thúc utterance; tốc độ quá khác vẫn gây mismatch.'),
('Có cần chia score cho T','Không cần cho argmax trong cùng utterance vì T dương và giống nhau giữa model. Chia T hữu ích khi so các utterance khác độ dài hoặc đặt ngưỡng reject. Nếu cộng LM, phép chia làm đổi thang tương đối và cần ghi rõ.'),
('Warping path và state path','DTW căn frame test với frame template, tối thiểu chi phí trên lưới hai chuỗi quan sát. Viterbi căn frame với state ẩn, tối đa xác suất transition/emission của model học từ nhiều mẫu. Cả hai dùng quy hoạch động và truy vết.'),
('Unknown và reject','Argmax trong 5 lớp luôn chọn một lớp, kể cả mọi score đều thấp. Đề xuất reject nếu max log-likelihood/frame &lt; τ hoặc gap/frame &lt; δ; có thể thêm model nền. Cần validation có lệnh đúng và OOV độc lập để chọn ngưỡng; chưa có nên chưa xác nhận hiệu quả.')]
for i,(q,a) in enumerate(answers,1):
    p(f'<b>{i}. {q}.</b> {a}',size=9.3)
footer(6)
c.save()
print(OUT)

# Raster hóa từng trang bằng PDFium bundled để kiểm tra hình ảnh.
import pypdfium2 as pdfium
qa=ROOT/'results/report_qa';qa.mkdir(exist_ok=True)
pdf=pdfium.PdfDocument(str(OUT))
assert len(pdf)==6
for i,page in enumerate(pdf):
    page.render(scale=1.4).to_pil().save(qa/f'page-{i+1}.png')
print('Rendered 6 pages for visual QA')
