"""Reference-compatible final-answer EM; tool evidence never earns reward."""
import re
import unicodedata


def terminal_text(text):
    if '<|im_start|>' in text:
        tail=text.rsplit('<|im_start|>',1)[1]
        if not tail.startswith('assistant\n'):
            return ''
        text=tail[len('assistant\n'):]
    return text.split('<|im_end|>',1)[0].split('<|endoftext|>',1)[0].strip()


def normalize_answer(text):
    lowered=text.lower()
    without_punctuation=''.join(c for c in lowered if not unicodedata.category(c).startswith('P'))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',without_punctuation).split())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    info=extra_info or {}
    # The agent records its final generated turn before tool context is merged.
    text=terminal_text(info.get('terminal_assistant_text',solution_str))
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    answers=re.findall(r'^[ \t]*Answer:[ \t]*(.*?)[ \t]*$',text,flags=re.IGNORECASE|re.MULTILINE)
    valid=(len(answers)==1 and bool(answers[0].strip()) and '<tool_call>' not in text
           and bool(lines) and bool(re.fullmatch(r'Answer:[ \t]*\S.*',lines[-1],flags=re.IGNORECASE)))
    if isinstance(ground_truth,dict):
        ground_truth=ground_truth.get('target',[])
    if isinstance(ground_truth,str):
        ground_truth=[ground_truth]
    references=list(ground_truth) if ground_truth is not None else []
    exact=valid and any(normalize_answer(answers[0])==normalize_answer(str(a)) for a in references)
    result={'score':float(exact) if valid else -0.1,'acc':float(exact),
            'exact_match':float(exact),'format_rate':float(valid),
            'search_calls':float(info.get('search_calls',0)),
            'search_successes':float(info.get('search_successes',0))}

    if info.get('autosearch_enabled'):
        from autosearch.process_reward import compute_process_rewards
        result.update(base_score=result['score'], process_reward=0.0,
                      quality_reward=0.0, efficiency_reward=0.0, probe_valid=0.0)
        answers_by_turn=info.get('autosearch_turn_answers', [])
        validity=info.get('autosearch_turn_valid', [])
        if (valid and info.get('autosearch_terminal_action') and
                info.get('autosearch_probe_ok') and len(answers_by_turn)==len(validity)
                and all(isinstance(answer,str) and answer.strip() for answer in answers_by_turn)):
            process=compute_process_rewards(answers_by_turn, references, bool(exact), [*validity, True])
            result.update({key:process[key] for key in ('process_reward','quality_reward','efficiency_reward')})
            result['probe_valid']=1.0
            result['first_correct_turn']=float(process['first_correct_turn'] if process['first_correct_turn'] is not None else -1)
            result['score']+=process['process_reward']
    return result
