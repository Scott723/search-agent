"""Score exactly one final-answer record per question, including failed runs."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

from search_r1.io import read_jsonl,write_json
from search_r1.reward import compute_score


def summarize(questions,trajectories):
    expected={str(row['id']):row for row in questions}
    counts=Counter(str(row['question_id']) for row in trajectories)
    if len(expected)!=len(questions) or set(counts)!=set(expected) or any(n!=1 for n in counts.values()):
        raise ValueError('Expected exactly one result per unique question ID.')
    groups=defaultdict(list);scores=[]
    for trajectory in trajectories:
        question=expected[str(trajectory['question_id'])]
        score=compute_score(question['data_source'],'',question['answers'],extra_info=trajectory)
        groups[question['data_source']].append(score);scores.append(score)
    def average(rows):
        return {'count':len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in
                ['exact_match','format_rate','search_calls','search_successes']}}
    if not scores:raise ValueError('No evaluation questions.')
    return {**average(scores),'by_source':{k:average(v) for k,v in groups.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--trajectories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(args.trajectories.glob('*.jsonl')) if args.trajectories.is_dir() else [args.trajectories]
    result=summarize(read_jsonl(args.questions),[row for p in paths for row in read_jsonl(p)])
    write_json(args.output,result)
    print(result)


if __name__=='__main__':main()
