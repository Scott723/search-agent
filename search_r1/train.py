"""Shared veRL GRPO runner for Search-R1 and AutoSearch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .io import read_jsonl,write_json


def milestones(start,stop,every):
    if start<0 or stop<=start or every<=0:raise ValueError('Invalid training range.')
    return sorted({s for s in [20,*range(every,stop+1,every),stop] if start<s<=stop})


def configuration(args):
    output=args.output.resolve()
    cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text(encoding='utf-8'))
    cfg['data'].update(train_files=[str(args.train_file.resolve())],val_files=[str(args.val_file.resolve())])
    cfg['actor_rollout_ref']['model'].update(path=str(args.model.resolve()),
        lora_rank=0 if args.base else 8,lora_adapter_path=None)
    rollout=cfg['actor_rollout_ref']['rollout']
    rollout.update(prompt_length=1024,response_length=15360)
    rollout['agent']['agent_loop_config_path']=str(output/'agent-loop.yaml')
    rollout['multi_turn']['tool_config_path']=str(output/'tools.yaml')
    evaluation=args.mode=='eval'
    if evaluation:rollout['n']=1
    cfg['reward']['custom_reward_function']={'path':str(Path(__file__).with_name('reward.py')),'name':'compute_score'}
    cfg['trainer'].update(experiment_name=args.method,total_training_steps=max(args.steps,1),
        default_local_dir=str(output/'checkpoints'),rollout_data_dir=str(output/'generations'),
        validation_data_dir=str(output/'validation'),val_before_train=evaluation,val_only=evaluation,
        test_freq=-1,resume_mode='resume_path' if args.resume else 'disable',
        resume_from_path=str(args.resume.resolve()) if args.resume else None)
    env={'SEARCH_R1_RUN_DIR':str(output),'VERL_USE_UV':'0',
         'AUTOSEARCH_ENABLED':'1' if args.method=='autosearch' and not evaluation else '0',
         'PYTHONPATH':str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')}
    cfg['ray_kwargs']['ray_init'].update(_temp_dir='/tmp/search-agent-ray',
        runtime_env={'py_executable':sys.executable,'env_vars':env})
    return cfg


def write_config(args):
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=configuration(args)
    tools={'tools':[{'class_name':'search_r1.tool.BM25SearchTool',
        'config':{'type':'native','url':args.retrieval_url.rstrip('/')+'/retrieve',
                  'tokenizer_path':str(args.model.resolve()),'topk':3,'max_tokens':1536},
        'tool_schema':{'type':'function','function':{'name':'search',
            'description':'Search Wikipedia passages using a concise English query.',
            'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}}]}
    for name,value in [('config.yaml',cfg),('tools.yaml',tools),
                       ('agent-loop.yaml',[{'name':'search_r1_agent','_target_':'search_r1.loop.SearchR1AgentLoop'}])]:
        (args.output/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    return cfg


def validate_checkpoint(path,mode,steps):
    if path is None:return
    try:step=int(path.name.removeprefix('global_step_'))
    except ValueError:raise ValueError('Checkpoint directory must be named global_step_<number>.') from None
    if mode=='train' and step>steps:raise ValueError('Checkpoint step exceeds the training target.')
    required=['actor/model_world_size_1_rank_0.pt']
    if mode=='train':required+=['data.pt','actor/optim_world_size_1_rank_0.pt','actor/extra_state_world_size_1_rank_0.pt']
    missing=[name for name in required if not (path/name).is_file()]
    if missing:raise ValueError('Incomplete checkpoint: '+', '.join(missing))


def execute(args):
    validate_checkpoint(args.resume,args.mode,args.steps)
    import ray
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    import verl
    from verl.trainer.main_ppo import main as verl_main
    from evaluate import summarize
    cfg=write_config(args)
    for path in [args.train_file,args.val_file]:
        if not path.is_file():raise FileNotFoundError(path)
    for key,value in cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars'].items():os.environ[key]=value
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'),version_base=None):
        defaults=compose(config_name='ppo_trainer')
    OmegaConf.set_struct(defaults,False)
    config=OmegaConf.merge(defaults,OmegaConf.create(cfg));OmegaConf.resolve(config)
    OmegaConf.save(config,args.output/'resolved.yaml')
    started=time.time()
    write_json(args.output/'status.json',{'state':'running','pid':os.getpid()})
    try:
        already_saved=(args.mode=='train' and args.resume and int(args.resume.name.removeprefix('global_step_'))==args.steps)
        if not already_saved:verl_main.__wrapped__(config)
        if args.mode=='eval':
            rows=[r for p in (args.output/'trajectories').glob('*.jsonl') for r in read_jsonl(p)]
            write_json(args.output/'summary.json',summarize(read_jsonl(args.val_file.with_suffix('.jsonl')),rows))
        write_json(args.output/'status.json',{'state':'completed','elapsed_s':time.time()-started})
    except BaseException as error:
        write_json(args.output/'status.json',{'state':'failed','error_type':type(error).__name__})
        raise
    finally:ray.shutdown()


def run(args):
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs={'method':args.method,'model':str(args.model.resolve()),
                'files':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train_file,args.val_file]}}
        stamp=args.output/'inputs.json'
        if stamp.exists() and json.loads(stamp.read_text())!=inputs:raise ValueError('Run inputs changed; use a new output directory.')
        write_json(stamp,inputs)
        start=int(args.resume.name.removeprefix('global_step_')) if args.resume else 0
        previous=args.resume
        for step in milestones(start,args.steps,args.eval_every):
            for mode in ['train','eval']:
                directory=args.output/f'{mode}{step}';directory.mkdir(exist_ok=True)
                status=directory/'status.json'
                if status.exists() and json.loads(status.read_text())['state']=='completed':
                    if mode=='train':previous=directory/f'checkpoints/global_step_{step}'
                    continue
                resume=previous
                latest=directory/'checkpoints/latest_checkpointed_iteration.txt'
                if mode=='train' and latest.exists():resume=directory/f'checkpoints/global_step_{int(latest.read_text())}'
                for name in ['trajectories','search','validation','generations']:
                    p=directory/name
                    if p.exists():p.rename(directory/f'{name}.previous-{time.time_ns()}')
                command=[sys.executable,'-u','-m','search_r1.train','--method',args.method,'--mode',mode,
                    '--model',str(args.model.resolve()),'--train-file',str(args.train_file.resolve()),
                    '--val-file',str(args.val_file.resolve()),'--steps',str(step),'--output',str(directory.resolve()),
                    '--retrieval-url',args.retrieval_url]
                if resume:command.extend(['--resume',str(resume.resolve())])
                write_json(args.output/'status.json',{'state':'running','phase':f'{mode}{step}'})
                with (directory/'stage.log').open('a',encoding='utf-8') as log:
                    result=subprocess.run(command,stdout=log,stderr=log)
                if result.returncode:
                    write_json(args.output/'status.json',{'state':'failed','phase':f'{mode}{step}'})
                    raise RuntimeError(f'{mode}{step} failed; inspect {directory}/stage.log')
                if mode=='train':previous=directory/f'checkpoints/global_step_{step}'
        write_json(args.output/'status.json',{'state':'completed'})


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['search-r1','autosearch'],default='search-r1')
    parser.add_argument('--mode',choices=['run','train','eval'],default='run')
    parser.add_argument('--model',type=Path,default=Path(os.environ.get('MODEL_PATH','models/Qwen3.5-4B')))
    parser.add_argument('--train-file',type=Path,default=Path('data/train5000.parquet'))
    parser.add_argument('--val-file',type=Path,default=Path('data/eval100.parquet'))
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--output',type=Path,default=Path('runs/search-r1'))
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--base',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--retrieval-url',default=os.environ.get('RETRIEVAL_URL','http://127.0.0.1:8000'))
    args=parser.parse_args()
    if args.base and (args.mode!='eval' or args.resume):parser.error('--base requires --mode eval without --resume')
    if args.mode=='eval' and not args.base and not args.resume:parser.error('Trained evaluation requires --resume')
    if args.dry_run:
        write_config(args);print('CONFIG_READY',args.output/'config.yaml')
    elif args.mode=='run':run(args)
    else:execute(args)


if __name__=='__main__':main()
