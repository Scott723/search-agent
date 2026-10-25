"""Validate the final train/evaluation JSONL files and write veRL parquet inputs."""
import argparse
from collections import Counter
from pathlib import Path
import re
import unicodedata

from .io import read_jsonl

SYSTEM_PROMPT='''You answer factual questions with help from a search tool.
Search when you need evidence. You may call search several times with concise English queries.
Call search exactly once per assistant turn. Wait for the tool result before making another search call.
When ready, end with exactly one non-empty line in this format:
Answer: <your short answer>
Do not call a tool and give the final answer in the same turn.'''


def question_key(text):
    return re.sub(r'\W+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()


def validate(train,evaluation):
    if Counter(r['data_source'] for r in train)!={'nq':2500,'hotpotqa':2500}:
        raise ValueError('Training requires NQ/HotpotQA: 2500 questions each.')
    if Counter(r['data_source'] for r in evaluation)!=dict.fromkeys(['nq','hotpotqa','bamboogle','triviaqa'],25):
        raise ValueError('Evaluation requires four sources with 25 questions each.')
    for rows in [train,evaluation]:
        if len({str(r['id']) for r in rows})!=len(rows):raise ValueError('Duplicate question IDs; prefix them with data source.')
        if len({question_key(r['question']) for r in rows})!=len(rows):raise ValueError('Duplicate questions.')
        if any(not r['question'].strip() or not isinstance(r['answers'],list) or not r['answers'] for r in rows):
            raise ValueError('Questions and answer lists must be nonempty.')
    if {question_key(r['question']) for r in train}&{question_key(r['question']) for r in evaluation}:
        raise ValueError('Training and evaluation questions overlap.')


def write_parquet(rows,path,split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    records=[{'data_source':r['data_source'],
              'prompt':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':r['question']}],
              'agent_name':'search_r1_agent','ability':'fact_qa',
              'reward_model':{'style':'rule','ground_truth':r['answers']},
              'extra_info':{'index':i,'question_id':str(r['id']),'question':r['question'],'split':split}}
             for i,r in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(records),path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,default=Path('data/train5000.jsonl'))
    parser.add_argument('--eval',type=Path,default=Path('data/eval100.jsonl'))
    args=parser.parse_args()
    train,evaluation=read_jsonl(args.train),read_jsonl(args.eval)
    validate(train,evaluation)
    write_parquet(train,args.train.with_suffix('.parquet'),'train')
    write_parquet(evaluation,args.eval.with_suffix('.parquet'),'eval')
    print('Ready: train=5000, eval=100')


if __name__=='__main__':main()
