"""Document/passage BM25 indexing and a shared retrieval API."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from threading import RLock


def split_passages(docid,text,words=256,overlap=64,max_chars=6000):
    if words<=0 or not 0<=overlap<words or max_chars<=0:
        raise ValueError('Invalid passage window or overlap.')
    spans=list(re.finditer(r'\S+',text))
    if not text:return
    for first in range(0,len(spans),words-overlap) if spans else [0]:
        last=min(first+words,len(spans))
        start=spans[first].start() if first else 0
        end=spans[last].start() if last<len(spans) else len(text)
        for offset in range(start,end,max_chars):
            stop=min(offset+max_chars,end)
            yield {'id':f'p:{docid}:{offset}:{stop}','parent_docid':str(docid),
                   'source_start':offset,'source_end':stop,'contents':text[offset:stop]}
        if last==len(spans):break


def build(corpus,output,mode='passage',threads=4):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    inputs=output/'input';inputs.mkdir()
    documents=passages=0
    with closing(sqlite3.connect(output/'corpus.sqlite')) as db, \
            Path(corpus).open(encoding='utf-8') as source, (inputs/'corpus.jsonl').open('w',encoding='utf-8') as stream:
        db.execute('CREATE TABLE documents(id TEXT PRIMARY KEY, contents TEXT NOT NULL)')
        for line in source:
            if not line.strip():continue
            row=json.loads(line);docid=str(row['id']);text=row['contents']
            db.execute('INSERT INTO documents VALUES(?,?)',(docid,text))
            chunks=split_passages(docid,text) if mode=='passage' else [{'id':docid,'contents':text}]
            for chunk in chunks:
                stream.write(json.dumps(chunk,ensure_ascii=False)+'\n');passages+=1
            documents+=1
            if documents%10000==0:db.commit()
        db.commit()
    subprocess.run([sys.executable,'-m','pyserini.index.lucene','--collection','JsonCollection',
                    '--input',str(inputs),'--index',str(output/'index'),
                    '--generator','DefaultLuceneDocumentGenerator','--threads',str(threads),
                    '--storePositions','--storeDocvectors','--storeRaw'],check=True)
    (output/'metadata.json').write_text(json.dumps({'mode':mode,'documents':documents,'entries':passages}),encoding='utf-8')
    print(f'Indexed {documents} documents / {passages} entries: {output}')


class BM25:
    def __init__(self,root,corpus_db=None):
        import jnius_config
        jnius_config.add_options('-Xmx4g','-XX:ActiveProcessorCount=4')
        from pyserini.pyclass import autoclass
        root=Path(root)
        index=root/'index' if (root/'index').is_dir() else root
        self.searcher=autoclass('io.anserini.search.SimpleSearcher')(str(index))
        self.searcher.set_bm25(.9,.4)
        self.database=Path(corpus_db) if corpus_db else root/'corpus.sqlite'
        if not self.database.exists():self.database=root.parent/'corpus.sqlite'
        self.lock=RLock()

    def document(self,docid):
        if self.database.exists():
            with closing(sqlite3.connect(f'file:{self.database}?mode=ro',uri=True)) as db:
                row=db.execute('SELECT contents FROM documents WHERE id=?',(str(docid),)).fetchone()
            if row:return {'id':str(docid),'contents':row[0]}
        with self.lock:
            doc=self.searcher.doc(str(docid))
            if doc is not None:
                raw=doc.get('raw')
                if raw:return json.loads(raw)
        raise KeyError(docid)

    def search(self,query,topk=3):
        with self.lock:
            rows=[]
            for hit in self.searcher.search(query,topk):
                raw=hit.lucene_document.get('raw') if hit.lucene_document is not None else None
                document=json.loads(raw) if raw else self.document(hit.docid)
                rows.append({'document':document,'score':float(hit.score)})
            return rows


def create_app(backend):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,Field
    class Request(BaseModel):
        queries:list[str]=Field(min_length=1,max_length=16)
        topk:int=Field(default=3,ge=1,le=10)
    app=FastAPI()
    @app.get('/health')
    def health():return {'ok':True,'backend':'bm25'}
    @app.post('/retrieve')
    def retrieve(request:Request):
        if any(not q.strip() or len(q)>2000 for q in request.queries):
            raise HTTPException(422,'Queries must contain 1–2000 characters.')
        return {'result':[backend.search(q,request.topk) for q in request.queries]}
    @app.get('/document')
    def document(docid:str):
        try:return backend.document(docid)
        except KeyError:raise HTTPException(404,'Document not found.') from None
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    index=commands.add_parser('index')
    index.add_argument('--corpus',required=True)
    index.add_argument('--output',required=True)
    index.add_argument('--mode',choices=['document','passage'],default='passage')
    index.add_argument('--threads',type=int,default=4)
    serve=commands.add_parser('serve')
    serve.add_argument('--index',required=True)
    serve.add_argument('--corpus-db')
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    if args.command=='index':build(args.corpus,args.output,args.mode,args.threads)
    else:
        import uvicorn
        uvicorn.run(create_app(BM25(args.index,args.corpus_db)),host=args.host,port=args.port)


if __name__=='__main__':main()
