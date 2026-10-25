"""CPU BM25 tool: raw retrieved passages, with an explicit token budget."""
import asyncio
import time

import httpx
from transformers import AutoTokenizer
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from .io import append_record


def format_documents(rows, tokenizer, max_tokens=1536):
    text='\n\n'.join(f'[{i}] Document {row["document"]["id"]}\n{row["document"]["contents"]}'
                      for i,row in enumerate(rows,1))
    ids=tokenizer.encode(text,add_special_tokens=False)
    return tokenizer.decode(ids[:max_tokens],skip_special_tokens=False)


class BM25SearchTool(BaseTool):
    def __init__(self,config,tool_schema):
        super().__init__(config,tool_schema)
        self.url=config.get('url','http://127.0.0.1:8000/retrieve')
        self.topk=int(config.get('topk',3))
        self.max_tokens=int(config.get('max_tokens',1536))
        self.tokenizer=AutoTokenizer.from_pretrained(config['tokenizer_path'],local_files_only=True)
        self.semaphore=asyncio.Semaphore(4)

    async def execute(self,instance_id,parameters,**kwargs):
        query=parameters.get('query') if isinstance(parameters,dict) else None
        started=time.monotonic()
        error=None
        rows=[]
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            error='invalid_query'
        else:
            try:
                async with self.semaphore, httpx.AsyncClient(timeout=60) as client:
                    response=await client.post(self.url,json={'queries':[query],'topk':self.topk})
                    response.raise_for_status()
                    rows=response.json()['result'][0]
                if not rows: error='no_results'
            except (httpx.HTTPError,KeyError,IndexError,ValueError,TypeError):
                error='retrieval_failed'
        text=(f'Search error: {error}. No evidence was retrieved.' if error else
              format_documents(rows,self.tokenizer,self.max_tokens))
        elapsed=time.monotonic()-started
        append_record('search',{'backend':'bm25','query':query,'ok':error is None,'error':error,
                      'latency_s':elapsed,'attempts':int(error!='invalid_query'),
                      'search_requests':0,'result_count':len(rows),'usage':{},
                      'document_ids':[row['document']['id'] for row in rows]})
        return ToolResponse(text=text),0.0,{'search_ok':int(error is None),
                    'search_latency_s':elapsed,'search_results':len(rows)}
