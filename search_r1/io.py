"""JSONL inputs and process-local trajectory records."""
import json
import os
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def append_record(category,record):
    root=os.environ.get('SEARCH_R1_RUN_DIR')
    if root:
        directory=Path(root)/category;directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'{os.getpid()}.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
