"""AutoSearch core adapted from bofusun/AutoSearch (Apache-2.0).

Source commit: 75ce1c5aba097ded06318a2b1347f02175452eec
Source: verl/trainer/main_ppo_format_turn_new78.py, RewardManager, lines 550-614.
We retain its nonnegative F1 gains, early-correct boundary and reward constants.
Process terms are summed for GRPO; the existing final reward remains separate.
"""
import asyncio
from collections import Counter
import re
import time
from uuid import uuid4

from search_r1.reward import normalize_answer, terminal_text


def compute_process_rewards(turn_answers, ground_truth, final_correct, turn_valid):
    """Search rounds followed by one terminal assistant action; zero-based turns."""
    count = len(turn_answers)
    if len(turn_valid) != count + 1:
        raise ValueError('Expected one validity flag per search plus terminal action.')
    rewards = [0.0] * (count + 1)
    if not count:
        return dict(process_reward=0.0, quality_reward=0.0, efficiency_reward=0.0,
                    first_correct_turn=None, turn_rewards=rewards)
    references = [normalize_answer(str(a)) for a in ground_truth]
    first, best, quality = None, 0.0, 0.0
    for turn, answer in enumerate(turn_answers):
        prediction = normalize_answer(answer)
        predicted = Counter(prediction.split())
        f1 = 0.0
        for reference in references:
            expected = Counter(reference.split())
            overlap = sum((predicted & expected).values())
            denominator = sum(predicted.values()) + sum(expected.values())
            f1 = max(f1, 2 * overlap / denominator if denominator else 0.0)
        gain = max(0.0, f1 - best)
        best = max(best, f1)
        rewards[turn] += gain
        quality += gain
        # Official code excludes the final information round from early-correct.
        if turn < count - 1 and prediction in references:
            first = turn
            break
    for turn, valid in enumerate(turn_valid):
        if not final_correct:
            if valid:
                rewards[turn] += 0.0125
        elif first is not None:
            if turn <= first and valid:
                rewards[turn] += 0.5 * (0.4 / (first + 1) - 0.05)
            else:
                break
        elif valid:
            rewards[turn] += 0.5 * (0.4 / count - 0.05)
    if first is not None:
        for turn in range(first + 1, len(rewards)):
            rewards[turn] = -0.05
    total = sum(rewards)
    return dict(process_reward=total, quality_reward=quality,
                efficiency_reward=total-quality, first_correct_turn=first,
                turn_rewards=rewards)


async def generate_turn_answer(server_manager, tokenizer, prompt_ids, request_id,
                               max_model_len, max_tokens=256, timeout=120):
    """Side generation from the visible prefix only; never accepts gold labels."""
    started = time.monotonic()
    result = {'answer': None, 'status': 'generation_error', 'output_tokens': 0}
    try:
        trajectory = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content':
             'Answer the original question using the search trajectory below. '
             'Do not search or call tools. Output only Answer: followed by a short answer.\n\n'
             'Search trajectory:\n' + trajectory},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False)
        result['prompt_tokens'] = len(ids)
        if len(ids) + max_tokens > max_model_len:
            result['status'] = 'context_limit'
            return result
        output = await asyncio.wait_for(server_manager.generate(
            request_id=f'{request_id}-answer-{uuid4().hex}', prompt_ids=ids,
            sampling_params={'temperature': 0.0, 'top_p': 1.0, 'max_tokens': max_tokens}), timeout)
        result['output_tokens'] = len(output.token_ids)
        text = terminal_text(tokenizer.decode(output.token_ids, skip_special_tokens=False))
        match = re.fullmatch(r'Answer:[ \t]*(\S[^\n]*)', text, flags=re.IGNORECASE)
        if len(output.token_ids) >= max_tokens:
            result['status'] = 'output_limit'
        elif match and '<tool_call>' not in text:
            result.update(answer=match.group(1).strip(), status='ok')
        else:
            result['status'] = 'invalid_format'
    except asyncio.TimeoutError:
        result['status'] = 'timeout'
    except Exception as error:
        result['error_type'] = type(error).__name__
    finally:
        result['elapsed_s'] = time.monotonic() - started
    return result
