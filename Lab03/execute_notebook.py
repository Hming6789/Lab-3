"""Chạy notebook thật qua kernel cùng Python; không cần nbclient/nbconvert."""
from pathlib import Path
import os
import sys
import time
import queue
import nbformat
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parent
runtime=ROOT/'results/kernel_runtime'
runtime.mkdir(parents=True,exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
os.environ['IPYTHONDIR']=str(runtime/'ipython')
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
nb=nbformat.read(ROOT/'Lab03.ipynb',as_version=4)
km=KernelManager(kernel_name='python3')
km.connection_file=str(runtime/'kernel-lab03.json')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
print('Starting Conda Jupyter kernel',flush=True)
km.start_kernel(cwd=str(ROOT))
client=km.client()
client.start_channels()
started=time.monotonic()
try:
    print('Waiting for kernel ready',flush=True)
    client.wait_for_ready(timeout=90)
    count=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':
            continue
        count+=1
        cell.outputs=[]
        msgid=client.execute(cell.source,store_history=True,allow_stdin=False,stop_on_error=True)
        deadline=time.monotonic()+1800
        failed=False
        while True:
            if time.monotonic()>deadline:
                raise TimeoutError(f'Cell {i} timeout')
            try:
                msg=client.get_iopub_msg(timeout=5)
            except queue.Empty:
                continue
            if msg.get('parent_header',{}).get('msg_id')!=msgid:
                continue
            kind=msg['header']['msg_type'];content=msg['content']
            if kind=='status' and content.get('execution_state')=='idle':
                break
            if kind=='execute_input':
                cell.execution_count=content['execution_count']
            elif kind=='stream':
                cell.outputs.append(nbformat.v4.new_output('stream',name=content['name'],text=content['text']))
            elif kind in ['display_data','execute_result']:
                args=dict(data=content['data'],metadata=content.get('metadata',{}))
                if kind=='execute_result': args['execution_count']=content['execution_count']
                cell.outputs.append(nbformat.v4.new_output(kind,**args))
            elif kind=='error':
                failed=True
                cell.outputs.append(nbformat.v4.new_output('error',ename=content['ename'],
                                   evalue=content['evalue'],traceback=content['traceback']))
        nbformat.write(nb,ROOT/'Lab03.ipynb')
        print(f'Cell {i} / execution {count}: {"FAILED" if failed else "OK"}',flush=True)
        if failed:
            raise RuntimeError(str(cell.outputs[-1]))
    nb.metadata['execution_verified']=dict(python=sys.executable,code_cells=count,
                                          elapsed_s=round(time.monotonic()-started,2))
    nbformat.validate(nb)
    nbformat.write(nb,ROOT/'Lab03.ipynb')
    print('NOTEBOOK ALL CELLS PASSED',flush=True)
finally:
    client.stop_channels()
    km.shutdown_kernel(now=True)
    # Dọn client trước interpreter shutdown để tránh lỗi __del__ của traitlets.
    del client
    del km
