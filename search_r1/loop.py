"""Native tool loop with authoritative terminal reward fields and audit traces."""
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
import json
import os

from .io import append_record
from autosearch.process_reward import generate_turn_answer


class SearchR1AgentLoop(ToolAgentLoop):
    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        agent_data.extra_fields['terminal_assistant_text'] = (
            '' if state == AgentState.PROCESSING_TOOLS else
            self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False))
        if os.environ.get('AUTOSEARCH_ENABLED') == '1':
            fields = agent_data.extra_fields
            fields['autosearch_enabled'] = True
            fields.setdefault('autosearch_turn_answers', [])
            fields.setdefault('autosearch_turn_valid', [])
            fields.setdefault('autosearch_probe_ok', True)
            fields['autosearch_terminal_action'] = (
                state != AgentState.PROCESSING_TOOLS and
                '<tool_call>' not in fields['terminal_assistant_text'])
        return state

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        fields = agent_data.extra_fields
        if (os.environ.get('AUTOSEARCH_ENABLED') == '1' and
                len(fields.get('autosearch_turn_valid', [])) >
                len(fields.get('autosearch_turn_answers', []))):
            if state == AgentState.GENERATING:
                probe = await generate_turn_answer(
                    self.server_manager, self.tokenizer, agent_data.prompt_ids,
                    agent_data.request_id, self.rollout_config.max_model_len)
            else:
                probe = {'answer': None, 'status': 'unmerged_context'}
            fields.setdefault('autosearch_turn_answers', []).append(probe['answer'])
            fields.setdefault('autosearch_probes', []).append(probe)
            fields['autosearch_probe_ok'] = fields.get('autosearch_probe_ok', True) and probe['status'] == 'ok'
        return state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        response, reward, metrics = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        fields = agent_data.extra_fields
        fields['search_calls'] = fields.get('search_calls', 0) + int(tool_call.name == 'search')
        fields['search_successes'] = fields.get('search_successes', 0) + int(metrics.get('search_ok', 0))
        if os.environ.get('AUTOSEARCH_ENABLED') == '1' and tool_call.name == 'search':
            try:
                query = ' '.join(json.loads(tool_call.arguments)['query'].lower().split())
            except (ValueError, TypeError, KeyError, AttributeError):
                query = ''
            queries = fields.setdefault('autosearch_queries', [])
            fields.setdefault('autosearch_turn_valid', []).append(
                bool(query) and query not in queries and bool(metrics.get('search_ok', 0)))
            queries.append(query)
        return response, reward, metrics

    async def run(self, sampling_params, priority=0, **kwargs):
        output = await super().run(sampling_params, priority=priority, **kwargs)
        for key in ('search_calls', 'search_successes'):
            output.extra_fields.setdefault(key, 0)
        info = kwargs.get('extra_info') or {}
        append_record('trajectories', {
            'question_id': str(info.get('question_id', '')),
            'question': str(info.get('question', '')),
            'data_source': str(kwargs.get('data_source', '')),
            'ground_truth': list((kwargs.get('reward_model') or {}).get('ground_truth', [])),
            'response': self.tokenizer.decode(output.response_ids, skip_special_tokens=False),
            **{key: value for key, value in output.extra_fields.items() if key.startswith('autosearch_')},
            'response_ids': output.response_ids,
            'response_mask': output.response_mask,
            'num_turns': output.num_turns,
            **{key: output.extra_fields[key] for key in (
                'terminal_assistant_text', 'search_calls', 'search_successes')},
        })
        return output
