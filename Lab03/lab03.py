"""Lab 3 CSE457: pipeline tái lập; không sửa dữ liệu gốc, không fit trên test."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import platform
import sys
import time
import warnings

# Cache JIT nằm trong dự án; không ghi vào thư viện Conda base.
_cache_root = Path(__file__).resolve().parent / 'results' / 'runtime_cache'
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(_cache_root / 'numba'))
os.environ.setdefault('MPLCONFIGDIR', str(_cache_root / 'matplotlib'))

import joblib
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from hmmlearn.hmm import GaussianHMM
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
WORDS = dict(zip(LABELS, ['mở', 'đóng', 'lên', 'xuống', 'dừng']))
SEED = 457
CONFIG = dict(sr=16000, n_mfcc=13, n_fft=512, win_length=400,
              hop_length=160, window='hamming', n_mels=26, preemphasis=.97,
              trim_top_db=30, cmvn='utterance', n_states=5, covariance='diag',
              n_iter=80, tol=1e-3, seed=SEED)


def setup():
    for folder in ['models', 'language_model', 'reports', 'results/feature_plots',
                   'results/viterbi_plots', 'results/confusion_matrices',
                   'results/experiment_tables', 'results/convergence']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    versions = {}
    for p in ['numpy', 'scipy', 'matplotlib', 'librosa', 'soundfile',
              'scikit-learn', 'hmmlearn', 'pandas', 'joblib', 'jupyter', 'ipykernel', 'jupyter_client']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'metapackage chưa cài; dùng ipykernel trong VS Code'
    environment = dict(python=sys.version, executable=sys.executable,
                       platform=platform.platform(), packages=versions, config=CONFIG)
    save_json(environment, ROOT / 'results/environment.json')
    return environment


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def table(df, name):
    df.to_csv(ROOT / 'results/experiment_tables' / f'{name}.csv', index=False,
              encoding='utf-8-sig')
    return df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def audit_data():
    """Kiểm định header, hash byte/PCM, clipping, metadata, split trước fit."""
    data = ROOT / 'dataset'
    paths = sorted(data.glob('*/*/*.wav'))
    if len(paths) != 40:
        raise ValueError(f'Cần đúng 40 WAV, tìm thấy {len(paths)}')
    metadata = pd.read_csv(data / 'metadata.csv')
    if len(metadata) != 40 or metadata.path.duplicated().any():
        raise ValueError('Metadata thiếu dòng hoặc trùng đường dẫn')
    if set(metadata.path) != {p.relative_to(data).as_posix() for p in paths}:
        raise ValueError('Tập đường dẫn metadata khác WAV thực tế')
    rows, fatal = [], []
    for p in paths:
        rel = p.relative_to(data).as_posix()
        split, label = p.parts[-3:-1]
        try:
            info = sf.info(p)
            y, sr = sf.read(p, always_2d=True, dtype='float64')
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            pcm_sha = hashlib.sha256(y.tobytes()).hexdigest()
            if not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
                raise ValueError('Âm thanh rỗng, không hữu hạn hoặc toàn im lặng')
            m = metadata.loc[metadata.path.eq(rel)].iloc[0]
            checks = [m.label == label, m.split == split, m.text == WORDS[label],
                      str(m.sha256) == sha, int(m.sample_rate_hz) == sr,
                      int(m.channels) == info.channels, int(m.bit_depth) == 16,
                      abs(float(m.duration_seconds) - len(y)/sr) < 1e-3]
            if not all(checks):
                fatal.append(f'Metadata không khớp: {rel}')
            if not p.stem.startswith(label + '_'):
                fatal.append(f'Tên file khác nhãn: {rel}')
            # Khoảng im lặng xấp xỉ theo ngưỡng 1% peak; không phải nhãn VAD chuẩn.
            active = np.flatnonzero(np.max(np.abs(y), axis=1) > .01*np.max(np.abs(y)))
            rows.append(dict(path=rel, split=split, label=label, sr=sr,
                             channels=info.channels, subtype=info.subtype,
                             duration_s=len(y)/sr, peak=float(np.abs(y).max()),
                             clipping_samples=int((np.abs(y) >= 32767/32768).sum()),
                             leading_s=active[0]/sr, trailing_s=(len(y)-1-active[-1])/sr,
                             sha256=sha, pcm_sha256=pcm_sha, metadata_ok=all(checks),
                             format_ok=sr == 16000 and info.channels == 1 and info.subtype == 'PCM_16'))
        except Exception as e:
            fatal.append(f'{rel}: {type(e).__name__}: {e}')
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'results/data_audit.csv', index=False, encoding='utf-8-sig')
    duplicates = []
    for key in ['sha256', 'pcm_sha256']:
        for _, g in df.groupby(key):
            if len(g) > 1:
                duplicates.append(dict(kind=key, paths=g.path.tolist(),
                                       cross_split=g.split.nunique() > 1))
    for split, count in [('train', 6), ('test', 2)]:
        for label in LABELS:
            if len(df.loc[df.split.eq(split) & df.label.eq(label)]) != count:
                fatal.append(f'Sai số lượng {split}/{label}')
    if any(d['cross_split'] for d in duplicates):
        fatal.append('Phát hiện dữ liệu trùng byte hoặc PCM giữa train/test')
    summary = dict(files=len(df), train=int(df.split.eq('train').sum()),
                   test=int(df.split.eq('test').sum()), duplicates=duplicates,
                   fatal_errors=fatal, format_repairs_needed=int((~df.format_ok).sum()),
                   total_clipping_samples=int(df.clipping_samples.sum()),
                   duration_min_s=float(df.duration_s.min()), duration_max_s=float(df.duration_s.max()),
                   silence_note='PDF đề xuất 0.2–0.5 s đầu/cuối; xem leading_s/trailing_s đo gần đúng.',
                   source='synthetic_tts / eSpeak Vietnamese; chưa xác minh nhãn bằng người nghe')
    save_json(summary, ROOT / 'results/data_audit_summary.json')
    if fatal:
        raise ValueError('; '.join(fatal))
    return df, summary


def repair_formats(destination=None):
    """Tạo bản sao đúng định dạng nếu cần; tuyệt đối không ghi đè dataset gốc."""
    destination = Path(destination or ROOT / 'dataset_normalized').resolve()
    source = (ROOT / 'dataset').resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Chọn thư mục đích nằm ngoài dataset gốc')
    for p in source.glob('*/*/*.wav'):
        y = read_audio(p)
        out = destination / p.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(out)
        sf.write(out, y, 16000, subtype='PCM_16')
    return destination


def read_audio(path):
    try:
        y, sr = sf.read(Path(path), dtype='float64', always_2d=True)
    except Exception as e:
        raise ValueError(f'Không đọc được WAV {path}: {e}') from e
    y = y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError(f'Âm thanh rỗng hoặc không hợp lệ: {path}')
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


def extract_array(y, dims=39, details=False):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError('Tín hiệu phải mono, hữu hạn và có năng lượng')
    original = y.copy()
    y = y / np.max(np.abs(y)) * .95
    y, interval = librosa.effects.trim(y, top_db=30, frame_length=400, hop_length=160)
    if len(y) < 512:
        raise ValueError('Utterance sau VAD quá ngắn (<32 ms)')
    emphasized = librosa.effects.preemphasis(y, coef=.97)
    c = librosa.feature.mfcc(y=emphasized, sr=16000, n_mfcc=13, n_fft=512,
                            win_length=400, hop_length=160, window='hamming',
                            n_mels=26, fmin=20, fmax=8000, center=True)
    width = min(9, c.shape[1] if c.shape[1] % 2 else c.shape[1]-1)
    if width < 3:
        raise ValueError('Không đủ frame để tính delta')
    d1 = librosa.feature.delta(c, width=width, order=1)
    d2 = librosa.feature.delta(c, width=width, order=2)
    raw = np.vstack([c, d1, d2]).T
    # CMVN từng utterance; không tổng hợp test để fit bất cứ thống kê nào.
    feat = (raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-8)
    feat = np.ascontiguousarray(feat[:, :dims], dtype='float64')
    assert dims in [13, 39] and feat.shape[1] == dims and np.isfinite(feat).all()
    return (feat, (original, y, c, d1, d2, interval)) if details else feat


def extract_feat(path, dims=39):
    return extract_array(read_audio(path), dims)


def feature_stage(audit):
    cache = {r.path: extract_feat(ROOT / 'dataset' / r.path) for r in audit.itertuples()}
    shapes = pd.DataFrame([dict(path=p, frames=len(f), dimensions=f.shape[1]) for p, f in cache.items()])
    table(shapes, 'feature_shapes')
    for label in ['mo', 'xuong']:
        p = f'train/{label}/{label}_01.wav'
        f, (original, trimmed, c, d1, d2, interval) = extract_array(read_audio(ROOT/'dataset'/p), details=True)
        fig, axes = plt.subplots(5, 1, figsize=(9, 10))
        axes[0].plot(np.arange(len(original))/16000, original)
        axes[0].axvspan(interval[0]/16000, interval[1]/16000, alpha=.15, color='green')
        axes[0].set(title=f'{WORDS[label]} — WAV gốc; vùng giữ lại', xlabel='Giây')
        axes[1].plot(np.arange(len(trimmed))/16000, trimmed)
        axes[1].set(title=f'Chuẩn hóa và cắt silence; feature {f.shape}', xlabel='Giây')
        for ax, values, name in zip(axes[2:], [c, d1, d2], ['MFCC 13', 'Delta 13', 'Delta² 13']):
            im = ax.imshow(values, origin='lower', aspect='auto', extent=[0, values.shape[1]*.01, 0, 13])
            ax.set(title=name, xlabel='Giây', ylabel='Hệ số')
            fig.colorbar(im, ax=ax)
        savefig(ROOT/'results/feature_plots'/f'{label}.png')
    return cache, shapes


def initial_topology(n):
    a = np.zeros((n, n))
    for i in range(n-1):
        a[i, i], a[i, i+1] = .7, .3
    a[-1, -1] = 1
    return a


def assert_topology(model):
    n = model.n_components
    allowed = initial_topology(n) > 0
    assert np.array_equal(model.startprob_, np.eye(n)[0]), 'Sai xác suất bắt đầu'
    assert np.all(model.transmat_[~allowed] == 0), 'Chuyển trạng thái bị cấm khác 0'
    assert np.isfinite(model.transmat_).all() and np.all(model.transmat_ >= 0)
    assert np.allclose(model.transmat_.sum(axis=1), 1)
    assert model.transmat_[-1, -1] == 1
    assert np.isfinite(model.means_).all() and np.isfinite(model.covars_).all()


def make_lr_hmm(sequences, n_states=5, covariance='diag', fixed=False):
    """Khởi tạo Gaussian bằng chia thời gian đều; cùng quy tắc cho E1–E3."""
    d = sequences[0].shape[1]
    if min(map(len, sequences)) < n_states:
        raise ValueError('Không đủ frame cho số state yêu cầu')
    groups = [np.vstack([np.array_split(f, n_states)[s] for f in sequences]) for s in range(n_states)]
    model = GaussianHMM(n_components=n_states, covariance_type=covariance, n_iter=80,
                        tol=1e-3, min_covar=1e-3, init_params='',
                        params='mc' if fixed else 'tmc', random_state=SEED,
                        implementation='log', covars_prior=.01 if covariance == 'diag' else .01*np.eye(d),
                        covars_weight=2 if covariance == 'diag' else d+2)
    model.startprob_ = np.eye(n_states)[0]
    model.transmat_ = initial_topology(n_states)
    model.means_ = np.vstack([g.mean(axis=0) for g in groups])
    if covariance == 'diag':
        model.covars_ = np.vstack([np.maximum(g.var(axis=0), 1e-3) for g in groups])
    else:
        model.covars_ = np.stack([np.cov(g, rowvar=False) + 1e-3*np.eye(d) for g in groups])
    return model, [len(g) for g in groups]


def fit_models(audit, cache, n_states=5, dims=39, covariance='diag', tag='baseline', persist=False):
    models, rows, histories = {}, [], []
    started = time.perf_counter()
    for label in LABELS:
        paths = audit.loc[audit.split.eq('train') & audit.label.eq(label), 'path'].tolist()
        assert len(paths) == 6 and all(p.startswith('train/') for p in paths)
        seq = [cache[p][:, :dims] for p in paths]
        x, lengths = np.vstack(seq), [len(f) for f in seq]
        m, counts = make_lr_hmm(seq, n_states, covariance)
        messages = io.StringIO()
        handler = logging.StreamHandler(messages)
        logger = logging.getLogger('hmmlearn')
        logger.addHandler(handler)
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter('always')
            try:
                m.fit(x, lengths)
                try:
                    assert_topology(m)
                    fixed = False
                except AssertionError:
                    m, _ = make_lr_hmm(seq, n_states, covariance, fixed=True)
                    m.fit(x, lengths)
                    assert_topology(m)
                    fixed = True
            finally:
                logger.removeHandler(handler)
        elapsed = time.perf_counter() - t
        history = list(m.monitor_.history)
        delta = history[-1] - history[-2] if len(history) > 1 else None
        # monitor.converged cũng True khi chạm giới hạn; phân biệt bằng delta thực.
        tolerance_met = delta is not None and 0 <= delta < m.tol
        status = 'tolerance' if tolerance_met else ('decreased' if delta is not None and delta < 0 else 'iteration_limit')
        train_score = sum(m.score(f) for f in seq) / sum(lengths)
        assert np.isfinite(train_score)
        rows.append(dict(config=tag, label=label, n_states=n_states, dimensions=dims,
                         covariance=covariance, files=6, frames=len(x), fit_s=elapsed,
                         iterations=len(history), monitor_converged=bool(m.monitor_.converged),
                         tolerance_met=tolerance_met, stop_reason=status, last_improvement=delta,
                         min_initial_state_frames=min(counts), train_score_per_frame=train_score,
                         topology_ok=True, transition_fixed_fallback=fixed,
                         warnings=' | '.join(str(w.message) for w in caught) + messages.getvalue()))
        histories.extend(dict(config=tag, label=label, iteration=i+1, log_likelihood=v) for i,v in enumerate(history))
        models[label] = m
        if persist:
            joblib.dump(m, ROOT/'models'/f'hmm_{label}.joblib')
            save_json(dict(label=label, word=WORDS[label], training_files=paths, lengths=lengths,
                           startprob=m.startprob_.tolist(), transmat=m.transmat_.tolist(), config=CONFIG),
                      ROOT/'models'/f'hmm_{label}.json')
    table(pd.DataFrame(rows), f'{tag}_training')
    table(pd.DataFrame(histories), f'{tag}_convergence')
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in pd.DataFrame(histories).groupby('label', sort=False):
        ax.plot(group.iteration, group.log_likelihood, label=label)
    ax.set(xlabel='Iteration', ylabel='Train log likelihood', title=f'Baum–Welch: {tag}')
    ax.legend()
    savefig(ROOT/'results/convergence'/f'{tag}.png')
    return models, pd.DataFrame(rows), time.perf_counter()-started


def acoustic_scores(f, models):
    scores = {}
    for label, m in models.items():
        value = float(m.score(f))
        if not np.isfinite(value):
            raise FloatingPointError(f'Score NaN/inf: {label}')
        scores[label] = value
    return scores


def recognize(wav_path, models):
    """Trả từ tiếng Việt, scores và xếp hạng; Forward để nhận dạng."""
    dims = next(iter(models.values())).means_.shape[1]
    scores = acoustic_scores(extract_feat(wav_path, dims), models)
    ranking = sorted(scores, key=scores.get, reverse=True)
    return WORDS[ranking[0]], scores, ranking


def cm_plot(truth, predicted, tag):
    cm = confusion_matrix(truth, predicted, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set(title=tag, xlabel='Dự đoán', ylabel='Nhãn thật')
    savefig(ROOT/'results/confusion_matrices'/f'{tag}.png')
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(ROOT/'results/confusion_matrices'/f'{tag}.csv', encoding='utf-8-sig')
    return cm


def evaluate(audit, models, cache=None, dims=39, tag='HMM_clean', transform=None):
    rows = []
    for r in audit.loc[audit.split.eq('test')].itertuples():
        started = time.perf_counter()
        if transform is not None:
            f = extract_array(transform(read_audio(ROOT/'dataset'/r.path)), dims)
        elif cache is not None:
            f = cache[r.path][:, :dims]
        else:
            f = extract_feat(ROOT/'dataset'/r.path, dims)
        s = acoustic_scores(f, models)
        ranking = sorted(s, key=s.get, reverse=True)
        pred, second = ranking[:2]
        rows.append(dict(path=r.path, truth=r.label, predicted=pred, predicted_word=WORDS[pred],
                         top2=second, correct=pred == r.label, frames=len(f),
                         score_top1=s[pred], score_top2=s[second], gap=s[pred]-s[second],
                         gap_per_frame=(s[pred]-s[second])/len(f),
                         true_score_per_frame=s[r.label]/len(f),
                         prediction_ms=(time.perf_counter()-started)*1000,
                         **{f'score_{k}':v for k,v in s.items()}))
    df = pd.DataFrame(rows)
    table(df, f'{tag}_predictions')
    cm_plot(df.truth, df.predicted, tag)
    save_json(classification_report(df.truth, df.predicted, labels=LABELS,
                                   output_dict=True, zero_division=0), ROOT/'results'/f'{tag}_classification_report.json')
    return df


def viterbi_stage(audit, cache, models):
    rows = []
    for label in ['mo', 'xuong']:
        p = audit.loc[audit.split.eq('test') & audit.label.eq(label), 'path'].iloc[0]
        f = cache[p]
        # Alignment với model nhãn thật, ghi rõ để không lẫn với decoding model top-1.
        logp, states = models[label].decode(f, algorithm='viterbi')
        forward = models[label].score(f)
        assert np.all(np.diff(states) >= 0) and np.all(np.diff(states) <= 1)
        assert logp <= forward + 1e-7 and states[0] == 0
        pd.DataFrame(dict(frame=np.arange(len(states)), seconds=np.arange(len(states))*.01,
                          state=states)).to_csv(ROOT/'results/viterbi_plots'/f'{label}_path.csv', index=False)
        for state in range(5):
            frames = int((states == state).sum())
            rows.append(dict(path=p, model=label, state=state, frames=frames,
                             duration_ms=frames*10, viterbi_logp=logp, forward_logp=forward))
        plt.figure(figsize=(8, 3))
        plt.step(np.arange(len(states))*.01, states, where='post')
        plt.yticks(range(5))
        plt.xlabel('Giây sau cắt silence (bước frame 10 ms)')
        plt.ylabel('State')
        plt.title(f'{p} — alignment model nhãn thật {label}')
        savefig(ROOT/'results/viterbi_plots'/f'{label}.png')
    return table(pd.DataFrame(rows), 'viterbi_occupancy')


def dtw_distance(a, b):
    """Chi phí Euclid trung bình trên đường tối ưu; chọn template gần nhất."""
    cost = cdist(a, b, metric='euclidean')
    cumulative, steps = librosa.sequence.dtw(C=cost, backtrack=True)
    return float(cumulative[-1, -1] / len(steps))


def dtw_stage(audit, cache, transform=None, tag='DTW_clean', dims=13):
    t = time.perf_counter()
    templates = [(r.label, cache[r.path][:, :dims]) for r in audit.loc[audit.split.eq('train')].itertuples()]
    prepare_s = time.perf_counter()-t
    rows = []
    for r in audit.loc[audit.split.eq('test')].itertuples():
        started = time.perf_counter()
        f = (extract_array(transform(read_audio(ROOT/'dataset'/r.path)), dims)
             if transform else extract_feat(ROOT/'dataset'/r.path, dims))
        distances = {label:min(dtw_distance(f, sample) for l,sample in templates if l == label) for label in LABELS}
        pred = min(distances, key=distances.get)
        rows.append(dict(path=r.path, truth=r.label, predicted=pred, correct=pred == r.label,
                         prediction_ms=(time.perf_counter()-started)*1000,
                         **{f'distance_{k}':v for k,v in distances.items()}))
    df = table(pd.DataFrame(rows), f'{tag}_predictions')
    cm_plot(df.truth, df.predicted, tag)
    return df, prepare_s


def parse_lexicon(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if not phones or word in result:
            raise ValueError(f'Dòng lexicon không hợp lệ: {line}')
        result[word] = phones
    return result


def language_stage(predictions):
    lm_dir = ROOT/'language_model'
    # Token là ĐƠN VỊ WHOLE-WORD GIẢ LẬP, không phải phiên âm tiếng Việt chuẩn.
    (lm_dir/'lexicon.txt').write_text(
        '# MINH HỌA định dạng; chưa có phone inventory của giảng viên.\n'
        '# Các token WW_* chỉ là đơn vị toàn từ giả lập, KHÔNG phải phoneme.\n'
        '# Cần giảng viên xác nhận/thay bằng inventory trước triển khai phone HMM.\n'
        'mở WW_MO\nđóng WW_DONG\nlên WW_LEN\nxuống WW_XUONG\ndừng WW_DUNG\n', encoding='utf-8')
    # Corpus tự xây dựng, không suy từ nhãn test. Corpus chỉ minh họa thứ tự lệnh.
    lines = ['mở lên dừng', 'mở lên dừng', 'mở lên đóng', 'đóng xuống dừng',
             'đóng xuống dừng', 'đóng xuống mở', 'lên dừng đóng', 'xuống dừng mở',
             'dừng mở lên', 'dừng đóng xuống', 'mở đóng', 'lên xuống']
    (lm_dir/'corpus.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    lexicon = parse_lexicon(lm_dir/'lexicon.txt')
    save_json(lexicon, lm_dir/'lexicon_parsed.json')
    inv = {v:k for k,v in WORDS.items()}
    counts1, counts2, outgoing = Counter(), Counter(), Counter()
    for line in lines:
        tokens = ['<s>']+[inv[w] for w in line.split()]+['</s>']
        counts1.update(tokens)
        for a,b in zip(tokens, tokens[1:]):
            counts2[a,b] += 1
            outgoing[a] += 1
    vocab = LABELS+['</s>']  # Không đưa <s> vào tập đích; có </s> để tổng xác suất = 1.
    k = .5
    def prob(prev, cur):
        return (counts2[prev,cur]+k)/(outgoing[prev]+k*len(vocab))
    for prev in ['<s>']+LABELS:
        assert np.isclose(sum(prob(prev,w) for w in vocab), 1)
    pd.DataFrame([dict(token=w,count=c) for w,c in sorted(counts1.items())]).to_csv(lm_dir/'unigram_counts.csv', index=False)
    pd.DataFrame([dict(previous=a,current=b,count=counts2[a,b],probability=prob(a,b))
                  for a in ['<s>']+LABELS for b in vocab]).to_csv(lm_dir/'bigram_counts.csv', index=False)
    rows = []
    # Không có context ghi âm: mọi context dưới đây là giả định minh họa, không báo accuracy LM thực.
    for r in predictions.itertuples():
        for prev in ['<s>']+LABELS:
            for weight in [0, .1, .3, 1.0]:
                acoustic = {w:getattr(r, f'score_{w}')/r.frames for w in LABELS}
                total = {w:acoustic[w]+weight*np.log(prob(prev,w)) for w in LABELS}
                pred = max(total,key=total.get)
                for w in LABELS:
                    rows.append(dict(path=r.path, truth=r.truth, previous=prev, lm_weight=weight,
                                     candidate=w, acoustic_score_per_frame=acoustic[w],
                                     lm_logprob=np.log(prob(prev,w)), total_score=total[w],
                                     prediction=pred, acoustic_prediction=r.predicted,
                                     changed=pred != r.predicted, hypothetical_context=True))
    scores = table(pd.DataFrame(rows), 'E6_LM_scores')
    decisions = scores.drop_duplicates(['path','previous','lm_weight'])
    summary = table(decisions.groupby('lm_weight').agg(trials=('prediction','size'),
                    changed_decisions=('changed','sum')).reset_index(), 'E6')
    table(decisions.loc[decisions.changed], 'E6_changed_decisions')
    close = predictions.loc[predictions.gap_per_frame < .5]
    illustration = []
    # Ví dụ điểm nhân tạo tách biệt rõ khỏi scores của WAV.
    for weight in [0, .1, .3, 1.0]:
        a = {'mo':-20.0, 'dong':-20.05, 'len':-20.1, 'xuong':-22, 'dung':-23}
        total = {w:a[w]+weight*np.log(prob('mo',w)) for w in LABELS}
        for w in LABELS:
            illustration.append(dict(source='ARTIFICIAL_SCORES_NOT_WAV', previous='mo', lm_weight=weight,
                                     candidate=w, acoustic=a[w], lm_logprob=np.log(prob('mo',w)),
                                     total=total[w], prediction=max(total,key=total.get)))
    table(pd.DataFrame(illustration), 'E6_artificial_illustration')
    save_json(dict(corpus_source='Tự xây dựng, không có context âm thanh thực', smoothing_k=k,
                   vocabulary=vocab, close_criterion='gap/frame < 0.5; chốt trước đánh giá',
                   close_test_files=close.path.tolist(), lexicon_status='minh họa WW_*; chờ inventory giảng viên'), lm_dir/'notes.json')
    return lexicon, summary


def experiment_stage(audit, cache, baseline, training, predictions, baseline_time):
    results, all_training = [], [training]
    configs = [('E1_3',3,39,'diag'), ('baseline',5,39,'diag'), ('E1_7',7,39,'diag'),
               ('E2_13',5,13,'diag'), ('E3_full',5,39,'full')]
    for tag,n,d,cov in configs:
        try:
            if tag == 'baseline':
                models, train, elapsed, pred = baseline, training, baseline_time, predictions
            else:
                models, train, elapsed = fit_models(audit, cache,n,d,cov,tag)
                pred = evaluate(audit,models,cache,d,tag)
                all_training.append(train)
            # train accuracy là chẩn đoán in-sample, không là đánh giá tổng quát hóa.
            train_rows = audit.loc[audit.split.eq('train')]
            correct = sum(max(acoustic_scores(cache[r.path][:,:d],models),
                              key=acoustic_scores(cache[r.path][:,:d],models).get) == r.label
                          for r in train_rows.itertuples())
            results.append(dict(config=tag, states=n, dimensions=d, covariance=cov,
                                accuracy=pred.correct.mean(), train_accuracy=correct/30,
                                fit_and_export_s=elapsed, fit_s=train.fit_s.sum(),
                                mean_iterations=train.iterations.mean(), tolerance_models=int(train.tolerance_met.sum()),
                                train_score_per_frame=train.train_score_per_frame.mean(),
                                test_true_score_per_frame=pred.true_score_per_frame.mean(), status='completed'))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
            results.append(dict(config=tag,states=n,dimensions=d,covariance=cov,status=f'failed: {e}'))
    combined = table(pd.DataFrame(results), 'configuration_results')
    table(pd.concat(all_training,ignore_index=True), 'all_training_diagnostics')
    table(combined.loc[combined.config.isin(['E1_3','baseline','E1_7'])], 'E1')
    table(combined.loc[combined.config.isin(['baseline','E2_13'])], 'E2')
    table(combined.loc[combined.config.isin(['baseline','E3_full'])], 'E3')
    # Ép một thread trong toàn bộ runner; thời gian là một lần đo, không benchmark thống kê.
    hmm_timed = evaluate(audit,baseline,None,39,'HMM_timed')
    # Warm-up DTW JIT trước đo để không phạt thời gian biên dịch lần đầu.
    dtw_distance(cache[next(iter(cache))][:,:13],cache[next(iter(cache))][:,:13])
    dtw, prep = dtw_stage(audit,cache)
    shared_dtw, prep39 = dtw_stage(audit,cache,tag='DTW_shared39',dims=39)
    comparison = table(pd.DataFrame([
        dict(method='HMM39',dimensions=39,accuracy=predictions.correct.mean(),preparation_s=training.fit_s.sum(),prediction_ms=hmm_timed.prediction_ms.mean(),train_per_class=6),
        dict(method='DTW13',dimensions=13,accuracy=dtw.correct.mean(),preparation_s=prep,prediction_ms=dtw.prediction_ms.mean(),train_per_class=6),
        dict(method='DTW39',dimensions=39,accuracy=shared_dtw.correct.mean(),preparation_s=prep39,prediction_ms=shared_dtw.prediction_ms.mean(),train_per_class=6)
    ]), 'E4')
    # So sánh HMM39 với DTW39 chỉ đổi recognizer; DTW13 là baseline MFCC Lab 2.
    paired = predictions[['path','truth','predicted']].merge(dtw[['path','predicted']],on='path',suffixes=('_HMM','_DTW'))
    table(paired,'E4_paired_cases')
    speed_rows = []
    for rate in [.8,1.2]:
        def stretch(y, rate=rate):
            return librosa.effects.time_stretch(y,rate=rate)
        hp = evaluate(audit,baseline,None,39,f'HMM_rate_{rate}',stretch)
        dp,_ = dtw_stage(audit,cache,stretch,f'DTW39_rate_{rate}',dims=39)
        speed_rows.extend([dict(rate=rate,method='HMM39',accuracy=hp.correct.mean()),
                           dict(rate=rate,method='DTW39',accuracy=dp.correct.mean())])
    table(pd.DataFrame(speed_rows), 'speed_robustness')
    def noise(y):
        # Seed riêng mỗi utterance để tái lập và không phụ thuộc thứ tự chạy.
        rng=np.random.default_rng(SEED+int(hashlib.sha256(y.tobytes()).hexdigest()[:8],16))
        return y+rng.normal(size=len(y))*np.sqrt(np.mean(y*y))/10  # SNR 20 dB trước trim
    robust = [dict(condition='clean',accuracy=predictions.correct.mean(),drop_pp=0)]
    for tag, transform in [('noise_20dB',noise),('volume_0.3',lambda y:y*.3)]:
        p = evaluate(audit,baseline,None,39,tag,transform)
        robust.append(dict(condition=tag,accuracy=p.correct.mean(),drop_pp=100*(predictions.correct.mean()-p.correct.mean())))
    table(pd.DataFrame(robust),'E5')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(combined.config,combined.accuracy*100)
    ax.set(ylabel='Accuracy (%)',ylim=(0,105),title='So sánh cấu hình; test chỉ dùng đánh giá')
    savefig(ROOT/'results/experiment_tables/configurations.png')
    return combined, comparison, pd.DataFrame(robust)


def reject_proposal(audit,cache,models):
    """Heuristic in-sample; chưa hiệu chuẩn OOV. Không báo FPR/recall unknown."""
    values,gaps=[],[]
    for r in audit.loc[audit.split.eq('train')].itertuples():
        f=cache[r.path]
        s=sorted(acoustic_scores(f,models).values(),reverse=True)
        values.append(s[0]/len(f)); gaps.append((s[0]-s[1])/len(f))
    proposal=dict(score_per_frame_threshold=float(np.quantile(values,.05)),
                  gap_per_frame_threshold=float(np.quantile(gaps,.05)),
                  rule='reject if top1/frame < score threshold OR gap/frame < gap threshold',
                  calibration='Heuristic phân vị 5% trên train in-sample; có thiên lệch lạc quan',
                  validation='Chưa có ngoài từ vựng, chưa xác minh unknown; cần validation độc lập + OOV')
    save_json(proposal,ROOT/'results/reject_proposal.json')
    return proposal


def score_plot(predictions):
    correct,wrong=[],[]
    for r in predictions.itertuples():
        for label in LABELS:
            (correct if label==r.truth else wrong).append(getattr(r,f'score_{label}')/r.frames)
    plt.figure(figsize=(7,4))
    plt.hist(correct,bins=10,alpha=.7,label='Model nhãn thật (10 scores)')
    plt.hist(wrong,bins=15,alpha=.6,label='Model nhãn khác (40 scores)')
    plt.xlabel('Log likelihood/frame');plt.ylabel('Số lượng');plt.legend()
    savefig(ROOT/'results/score_distribution.png')


def main():
    with threadpool_limits(limits=1):
        environment=setup()
        audit,summary=audit_data()
        print('DATA AUDIT',summary,flush=True)
        cache,shapes=feature_stage(audit)
        print('A completed',shapes.head(),flush=True)
        models,training,elapsed=fit_models(audit,cache,persist=True)
        print('B-C completed',training.to_string(index=False),flush=True)
        predictions=evaluate(audit,models,cache)
        predictions.to_csv(ROOT/'results/predictions.csv',index=False,encoding='utf-8-sig')
        table(predictions.head(3),'likelihood_examples')
        occupancy=viterbi_stage(audit,cache,models)
        print('D completed: clean accuracy',predictions.correct.mean(),flush=True)
        configurations,comparison,robustness=experiment_stage(audit,cache,models,training,predictions,elapsed)
        lexicon,lm=language_stage(predictions)
        rejection=reject_proposal(audit,cache,models)
        score_plot(predictions)
        summary_run=dict(status='completed',clean_accuracy=float(predictions.correct.mean()),
                         clean_correct=int(predictions.correct.sum()),test_files=10,
                         final_configuration='Preset PDF: 5 states, 39D, diag; NOT chosen on test',
                         dataset='eSpeak Vietnamese TTS only',speaker_mismatch='Chưa thực hiện',
                         lexicon='Illustrative WW units; instructor phone inventory unavailable',
                         lm_context='Hypothetical only; no recorded previous word',
                         failed_experiments=configurations.loc[configurations.status.ne('completed'),'config'].tolist())
        save_json(summary_run,ROOT/'results/run_summary.json')
        print('E-H / E1-E6 completed',flush=True)
        print(configurations.to_string(index=False),flush=True)
        print(comparison.to_string(index=False),flush=True)
        print(robustness.to_string(index=False),flush=True)
        return summary_run


if __name__=='__main__':
    main()
