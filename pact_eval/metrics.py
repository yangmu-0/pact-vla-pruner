"""Explicit FLOP/latency contracts; no model imports on the planning path."""
import statistics
import math
from collections import defaultdict

METRIC_NOTES = {
    'Acc.(%)': 'Explicit reproduction convention: 100 * mean_s(success_rate_method_s / success_rate_native_vanilla_s), four complete suites. The paper calls Acc relative accuracy but gives no aggregation equation; some printed values cannot be reconstructed from its four displayed suite rates. Also export ratio-of-means and matched-backend versions.',
    'FLOPs(T)': 'Operational repository convention: first LLM pass sum batch*(4*n*d^2+2*n^2*d+3*n*d*m)/1e12, MAC=1. Observed layer shapes, gated Llama FFN. Matches existing cache code and is close to Table1 baseline scale. Excludes vision, projector, action head and later decode passes; not total policy-call FLOPs.',
    'FLOPs_paper_Eq9(T)': 'Literal Appendix A.2 Eq.9: same first-pass formula but FFN coefficient=2. Paper equation differs from installed code. Kept separately; do not claim the undocumented table profiler is recovered exactly.',
    'FLOPs_profiled_call(T)': 'Sampled torch.profiler operation FLOPs over get_action, MAC=2; fused SDPA forward added from recorded Q/K/V shapes. Includes dispatched vision/projector/LLM decode/head/matmul selection, omits uncounted elementwise/softmax/sort and TensorFlow preprocessing. Dense attention arithmetic, not hardware instructions. Cold/warm sample summaries separate.',
    'Latency(ms)': 'CUDA-synchronized get_action wall time; first N global calls and FLOP-profiled calls excluded. Includes policy preprocessing, excludes model loading and simulator. OpenVLA: per action; OFT: per 8-action chunk. Online trajectories, not fixed-input replay.',
    'aggregation': 'Both first-pass FLOP formulas use all policy calls including cold/warmup. Per-suite latency averages eligible calls. Table averages the four suite means equally; does not silently pool partial suites.',
}


def layer_cost(batch, n, d, m, ffn_coefficient=2):
    return batch*(4*n*d*d + 2*n*n*d + ffn_coefficient*n*d*m)


def prefill_cost(visits):
    if set(visits)!=set(range(32)) or any(not v for v in visits.values()):
        raise ValueError('FLOP audit requires all 32 LLM layers.')
    first=[visits[i][0] for i in range(32)]
    return {'llm_prefill_eq9_T':sum(layer_cost(*v,2) for v in first)/1e12,
            'llm_prefill_gated_T':sum(layer_cost(*v,3) for v in first)/1e12,
            'llm_forward_count':len(visits[0]),
            'llm_layer_visit_counts':[len(visits[i]) for i in range(32)]}


def should_profile(episode_call, interval=50):
    return episode_call in (1,5) or (interval>0 and episode_call%interval==0)


def average(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize_calls(calls):
    profiled=[c for c in calls if c.get('profiled_flops')]
    return dict(
        flops_eq9_prefill_T=average([c.get('llm_prefill_eq9_T') for c in calls]),
        flops_gated_prefill_T=average([c.get('llm_prefill_gated_T') for c in calls]),
        flops_profiled_call_T=average([c.get('profiled_call_flops_T') for c in profiled]),
        flops_profiled_cold_T=average([c.get('profiled_call_flops_T') for c in profiled if c.get('episode_call')==1]),
        flops_profiled_warm_T=average([c.get('profiled_call_flops_T') for c in profiled if c.get('episode_call',0)>1]),
        flops_profiled_samples=len(profiled),
        flops_profiled_task_ids=sorted({c['task_id'] for c in profiled}),
        flops_formula_calls=sum(c.get('llm_prefill_eq9_T') is not None for c in calls),
        latency_profiled_calls_excluded=len(profiled),
        observed_llm_forward_counts=sorted({c['llm_forward_count'] for c in calls if c.get('llm_forward_count') is not None}))


class RuntimeFlopCounter:
    """CPU dispatch profiling sees nested inference_mode (TorchDispatchMode may not).

    No module hooks/cloning, no action replay and no changes to inference modes.
    torch.profiler omits fused SDPA FLOPs; add those once using its real shapes.
    """
    def __enter__(self):
        import torch
        self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            record_shapes=True,with_flops=True)
        self.profiler.__enter__()
        self.counts=defaultdict(int)
        return self

    def __exit__(self,*exc):
        self.profiler.__exit__(*exc)
        fused={'aten::_scaled_dot_product_flash_attention','aten::_scaled_dot_product_efficient_attention',
               'aten::_scaled_dot_product_flash_attention_for_cpu'}
        for event in self.profiler.events():
            count=int(event.flops or 0)
            if count:
                self.counts[event.name]+=count
            elif event.name in fused:
                q,k,v=event.input_shapes[:3]
                if len(q)!=4 or len(k)!=4 or len(v)!=4:
                    raise ValueError(f'Unrecognized fused attention shapes: {event.input_shapes}')
                self.counts[event.name+'[shape_formula]']+=2*math.prod(q[:-2])*q[-2]*k[-2]*(q[-1]+v[-1])

    def get_total_flops(self):
        return sum(self.counts.values())

    def get_flop_counts(self):
        return {'Global':dict(self.counts)}


def profile_counter():
    return RuntimeFlopCounter()
