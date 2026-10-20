"""Kiểm chứng tích hợp sản phẩm đã chạy, không fit lại hay thay đổi model."""
from pathlib import Path
import json
import hashlib
import nbformat
import numpy as np
import pandas as pd
import joblib
import lab03 as lab

ROOT=Path(__file__).resolve().parent
checks={}
nb=nbformat.read(ROOT/'Lab03.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert len(code)==7
assert [c.execution_count for c in code]==list(range(1,8))
assert not any(o.output_type=='error' for c in code for o in c.outputs)
assert 'execution_verified' in nb.metadata
checks['notebook_sequential_cells']=len(code)
checks['notebook_images']=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
assert checks['notebook_images']>=9

audit,summary=lab.audit_data()
checks['audio_files']=len(audit)
checks['cross_split_duplicates']=len(summary['duplicates'])
models={label:joblib.load(ROOT/'models'/f'hmm_{label}.joblib') for label in lab.LABELS}
pred=pd.read_csv(ROOT/'results/predictions.csv')
assert len(pred)==10 and pred.path.nunique()==10
assert np.isfinite(pred.filter(regex='^score_').to_numpy()).all()
assert pred.predicted.tolist()==pred.filter(regex='^score_(mo|dong|len|xuong|dung)$').idxmax(axis=1).str.replace('score_','').tolist()
for label,model in models.items():
    lab.assert_topology(model)
    metadata=json.loads((ROOT/'models'/f'hmm_{label}.json').read_text(encoding='utf-8'))
    assert len(metadata['training_files'])==6
    assert all(p.startswith(f'train/{label}/') for p in metadata['training_files'])
    assert not set(metadata['training_files'])&set(pred.path)
    for r in pred.itertuples():
        actual=model.score(lab.extract_feat(ROOT/'dataset'/r.path))
        assert np.isclose(actual,getattr(r,f'score_{label}'),rtol=1e-10,atol=1e-7)
checks['saved_models_and_50_scores']=True
checks['clean_accuracy']=float(pred.correct.mean())
configs=pd.read_csv(ROOT/'results/experiment_tables/configuration_results.csv')
assert len(configs)==5 and configs.status.eq('completed').all()
for n in range(1,7):
    p=ROOT/'results/experiment_tables'/f'E{n}.csv'
    assert p.exists() and len(pd.read_csv(p))>0
checks['experiments_completed']=['E1','E2','E3','E4','E5','E6']
paired=pd.read_csv(ROOT/'results/experiment_tables/E4_paired_cases.csv')
dtw39=pd.read_csv(ROOT/'results/experiment_tables/DTW_shared39_predictions.csv')
assert paired.path.is_unique and set(paired.path)==set(pred.path)==set(dtw39.path)
paired=paired.set_index('path')
assert paired.predicted_HMM.equals(pred.set_index('path').predicted.reindex(paired.index).rename('predicted_HMM'))
assert paired.predicted_DTW39.equals(dtw39.set_index('path').predicted.reindex(paired.index).rename('predicted_DTW39'))
checks['E4_pairs_use_DTW39']=True
for label in ['mo','xuong']:
    path=pd.read_csv(ROOT/'results/viterbi_plots'/f'{label}_path.csv')
    assert path.state.iloc[0]==0 and path.state.diff().dropna().isin([0,1]).all()
checks['viterbi_paths']=2
bigram=pd.read_csv(ROOT/'language_model/bigram_counts.csv')
assert np.allclose(bigram.groupby('previous').probability.sum(),1)
checks['bigram_normalized']=True
assert len(lab.parse_lexicon(ROOT/'language_model/lexicon.txt'))==5
checks['lexicon_parser']=True
# Bản sao WAV phải vẫn khớp bộ gốc, nếu đang chạy tại máy bàn giao.
original=ROOT.parent/'dataset'
if original.exists():
    for r in audit.itertuples():
        assert hashlib.sha256((original/r.path).read_bytes()).hexdigest()==r.sha256
    checks['original_wavs_unchanged']=True
checks['report_exists']=(ROOT/'reports/Bao_cao_Lab03_HMM.pdf').is_file()
assert checks['report_exists']
lab.save_json(checks,ROOT/'results/verification.json')
print(json.dumps(checks,ensure_ascii=False,indent=2))
