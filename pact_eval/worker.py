"""One isolated backend process. Existing evaluators own all simulation logic."""
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
from contextlib import nullcontext

# A run may execute its immutable worker/package snapshot. Prefer that package
# over later edits to the workspace; backend libraries still come from PYTHONPATH.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pact_eval.metrics import METRIC_NOTES, prefill_cost, profile_counter, should_profile, summarize_calls


class VerificationComplete(BaseException):
    """Do not let upstream episode exception handlers swallow a smoke stop."""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def episode_counts(out):
    outcomes, errors = [], []
    for file in sorted((out/'internal_logs').glob('*.txt')):
        text = file.read_text(errors='replace')
        outcomes.extend(re.findall(r'^Success: (True|False)\s*$', text, re.M))
        errors.extend(re.findall(r'^.*(?:Episode error:|Exception caught:).+$', text, re.M))
    return len(outcomes), outcomes.count('True'), errors


def expected_native_budgets(job, visual_tokens):
    """Return valid observed budgets for fixed or adaptive native pruning."""
    if job.get('ratio_semantics') == 'adaptive_retention':
        rates = tuple(float(value.strip()) for value in
                      job['settings']['pact_budget_rates'].split(','))
        expected = {round(visual_tokens * rate) for rate in rates}
        if job['model'] == 'oft':
            expected.update(2 * round(visual_tokens / 2 * rate) for rate in rates)
        return expected
    expected = {round(visual_tokens * (1 - job['ratio']))}
    if job['model'] == 'oft':
        expected.add(2 * round(visual_tokens / 2 * (1 - job['ratio'])))
    if job['strategy'] == 'vla-pruner':
        expected.add(visual_tokens)  # preserved temporal warmup/fallback
    return expected


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    root, out = Path(job['root']), Path(job['output'])
    result = dict(state='RUNNING', mode=job['mode'], episodes=0, successes=0, success_rate=None)
    calls, layer_lengths, loaded, adapter = [], {}, [], None
    environments = []
    tracking = {'episode':0,'episode_call':0,'task_id':None}
    layer_visits = {}
    policy_errors, budget_state = [], None
    started = time.time()
    fields = ['call', 'policy_ms', 'model_ms', 'cold', 'visual_kept', 'warm_sample']
    fields += ['task_id','episode','episode_call','profiled_flops','llm_prefill_eq9_T',
               'llm_prefill_gated_T','llm_forward_count','profiled_call_flops_T']
    csvfile = (out/'policy_timing_calls.csv').open('w', newline='', buffering=1)
    writer = csv.DictWriter(csvfile, fields)
    writer.writeheader()
    audits = (out/'token_audit.jsonl').open('w', buffering=1)
    flopfile = (out/'flops_profile_samples.jsonl').open('w',buffering=1)
    collect_flops = job.get('collect_flops',False)

    def report(state=None):
        if state:
            result['state'] = state
        episodes, successes, errors = episode_counts(out)
        warm = [c for c in calls if c['warm_sample']]
        result.update(episodes=episodes, successes=successes,
            success_rate=successes/episodes if episodes and job['mode']=='eval' else None,
            successful_calls=len(calls), measured_calls=len(warm),
            policy_ms_mean=statistics.mean(c['policy_ms'] for c in warm) if warm else None,
            model_ms_mean=statistics.mean(c['model_ms'] for c in warm) if warm else None,
            policy_ms_median=statistics.median(c['policy_ms'] for c in warm) if warm else None,
            all_calls_policy_ms_mean=statistics.mean(c['policy_ms'] for c in calls) if calls else None,
            warmup_calls_excluded=job['warmup_calls'], episode_errors=errors,
            policy_errors=policy_errors, wall_seconds=time.time()-started,
            observed_visual_kept=sorted(set(c['visual_kept'] for c in calls if c['visual_kept'] is not None)),
            latency_definition='CUDA-synchronized get_action; model=predict_action; excludes model loading and simulator. OFT unit=8-action chunk, OpenVLA unit=single action. Online trajectories, not fixed-input microbenchmark.',
            divprune_audit=adapter.summary() if adapter else None)
        if collect_flops:
            result.update(summarize_calls(calls),metric_definitions=METRIC_NOTES)
        save(out/'result.json', result)
        csvfile.flush()
        return result

    try:
        import numpy as np
        import torch
        import transformers
        evaluator_path = Path(job['cwd'])/'experiments/robot/libero/run_libero_eval.py'
        evaluation = load_module('pact_backend_evaluation', evaluator_path)
        original_env = evaluation.get_libero_env
        def get_env(*args, **kwargs):
            value = original_env(*args, **kwargs)
            environments.append(value[0])
            reset = value[0].reset
            def tracked_reset(*a, **kw):
                tracking['episode']+=1
                tracking['episode_call']=0
                return reset(*a, **kw)
            value[0].reset=tracked_reset
            return value
        evaluation.get_libero_env = get_env
        selected = job['task_ids']
        original_factories = evaluation.benchmark.get_benchmark_dict

        class SelectedSuite:
            def __init__(self, base):
                self.base = base
                self.n_tasks = len(selected)
                if any(i >= base.n_tasks for i in selected):
                    raise ValueError('Task ID out of bounds for installed LIBERO.')
                save(out/'tasks.json', [{'id':i, 'name':base.get_task(i).name,
                    'language':base.get_task(i).language} for i in selected])
            def get_task(self, index):
                tracking['task_id']=selected[index]
                return self.base.get_task(selected[index])
            def get_task_init_states(self, index):
                return self.base.get_task_init_states(selected[index])
            def __getattr__(self, name):
                return getattr(self.base, name)

        def factories():
            return {name:(lambda *a, _factory=factory, **kw: SelectedSuite(_factory(*a, **kw)))
                    for name, factory in original_factories().items()}
        evaluation.benchmark.get_benchmark_dict = factories
        if not job['save_video'] and hasattr(evaluation, 'save_rollout_video'):
            evaluation.save_rollout_video = lambda *a, **kw: None
        elif job['save_video'] and hasattr(evaluation, 'save_rollout_video'):
            original_video = evaluation.save_rollout_video
            def save_video(*args, **kwargs):
                previous = Path.cwd()
                try:
                    # Video helpers use relative output paths. Keep all new
                    # artifacts inside this attempt, away from historical runs.
                    os.chdir(out)
                    return original_video(*args, **kwargs)
                finally:
                    os.chdir(previous)
            evaluation.save_rollout_video = save_video

        # The native OFT source registers its current local model classes. Avoid
        # silently importing stale checkpoint-supplied Python files via HF cache.
        if job['model']=='oft' and job['backend_kind'] != 'vla-cache':
            from experiments.robot import openvla_utils
            original_auto = openvla_utils.AutoModelForVision2Seq
            class LocalAuto:
                @staticmethod
                def register(*args, **kwargs):
                    return original_auto.register(*args, **kwargs)
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    kwargs.update(trust_remote_code=False, local_files_only=True)
                    return original_auto.from_pretrained(*args, **kwargs)
            openvla_utils.AutoModelForVision2Seq = LocalAuto

        if job['strategy']=='vla-cache':
            cache = load_module('pact_fixed_cache', root/'vendor/vla-cache/vlacache_fixed_budget.py')
            budget_state = cache.install(job['ratio'])

        original_get_model = evaluation.get_model
        last_model_times = []

        def get_model(cfg):
            nonlocal adapter
            model = original_get_model(cfg)
            loaded.append(model)
            if job['backend_kind']=='divprune':
                div = load_module('pact_divprune_adapter', root/'vendor/vla-pruner/divprune_table1_20260830/divprune_adapter.py')
                div.OFFICIAL = root/'vendor/divprune-official/LLaVA/llava/model/llava_arch.py'
                adapter = div.Adapter(model, job['model'], job['ratio'], out)
                if job['model']=='oft':
                    # Upstream's metrics otherwise report a FastV-only budget
                    # even when the DivPrune pre-LLM adapter is active.
                    evaluation.get_pruning_call_metrics = lambda cfg, m: dict(
                        visual_tokens_before=512, visual_tokens_kept=round(512*(1-job['ratio'])),
                        flop_ratio=float('nan'), budget_key=str(round(512*(1-job['ratio']))))
            llama_model = model.language_model.model
            layers = llama_model.layers
            if job['strategy']=='pact-vla':
                original_pruning_indices = llama_model._fastv_pruning_indices
                def traced_pruning_indices(*args, **kwargs):
                    keep_indices, score_info = original_pruning_indices(*args, **kwargs)
                    # Retain GPU tensors only by reference here.  Conversion for JSON
                    # happens after the timed policy/model regions, so tracing does not
                    # perturb the latency measurement or pruning decision.
                    model._pact_raw_process_trace = (keep_indices.detach(), score_info)
                    return keep_indices, score_info
                llama_model._fastv_pruning_indices = traced_pruning_indices
            for index, layer in enumerate(layers):
                def before_layer(module, args, kwargs, index=index):
                    hidden = args[0] if args else kwargs['hidden_states']
                    if collect_flops:
                        layer_visits.setdefault(index,[]).append((int(hidden.shape[0]),int(hidden.shape[1]),
                            int(hidden.shape[2]),int(module.mlp.up_proj.out_features)))
                    if hidden.shape[1] > 1 and index not in layer_lengths:
                        layer_lengths[index] = int(hidden.shape[1])
                layer.register_forward_pre_hook(before_layer, with_kwargs=True)
            predict = model.predict_action
            def timed_predict(*args, **kwargs):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                value = predict(*args, **kwargs)
                torch.cuda.synchronize()
                last_model_times.append((time.perf_counter()-begin)*1000)
                return value
            model.predict_action = timed_predict
            sources = {str(evaluator_path), inspect.getfile(type(model)),
                       inspect.getfile(type(layers[0])), str(Path(__file__))}
            save(out/'environment.json', dict(python=sys.version, executable=sys.executable,
                torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                transformers_path=transformers.__file__, gpu=torch.cuda.get_device_name(0),
                model_class=str(type(model)), attention_classes=sorted({type(l.self_attn).__name__ for l in layers}),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(sources)},
                flags={k:getattr(model,k,None) for k in ('use_fastv','sparsevlm','use_temporal','use_text_vision_selection','use_prefil_attention','fastv_k','fastv_r')},
                settings=vars(cfg), migration_root=str(root),
                implementation='local_fixed_reuse_v2' if job['strategy']=='vla-cache' else 'preserved_local_sources'))
            return model
        evaluation.get_model = get_model
        original_action = evaluation.get_action

        def get_action(*args, **kwargs):
            layer_lengths.clear()
            layer_visits.clear()
            last_model_times.clear()
            if loaded and job['strategy']=='pact-vla':
                loaded[-1]._pact_raw_process_trace = None
            tracking['episode_call']+=1
            profiled=collect_flops and should_profile(tracking['episode_call'],job.get('flops_sample_interval',50))
            counter=profile_counter() if profiled else None
            if budget_state is not None:
                budget_state['selection_records'].clear()
                budget_state['stage_fractions'] = None
            cold = kwargs.get('last_caches') is None
            torch.cuda.synchronize()
            begin = time.perf_counter()
            try:
                with counter if counter is not None else nullcontext():
                    value = original_action(*args, **kwargs)
                torch.cuda.synchronize()
                policy_ms = 1000*(time.perf_counter()-begin)
                action = np.asarray(value[0])
                expected_shape = (7,) if job['model']=='openvla' else (8,7)
                if action.shape != expected_shape or not np.isfinite(action).all():
                    raise ValueError(f'Invalid action: shape {action.shape}, expected {expected_shape}, finite={np.isfinite(action).all()}')
                if not last_model_times:
                    raise ValueError('predict_action timing hook was not reached.')
                n = 256 if job['model']=='openvla' else 512
                if len(layer_lengths) != 32:
                    raise ValueError(f'Expected 32 prefill layers, got {layer_lengths}')
                kept = n - (layer_lengths[0]-layer_lengths[31])
                if adapter:
                    kept = int(round(n*(1-job['ratio'])))
                if not 0 < kept <= n:
                    raise ValueError(f'Invalid visual budget {kept}')
                if job['strategy']=='vanilla' and kept!=n:
                    raise ValueError(f'Vanilla unexpectedly changed token count: {kept} != {n}')
                if job['backend_kind']=='native' and job['strategy']!='vanilla':
                    expected = expected_native_budgets(job, n)
                    if kept not in expected:
                        raise ValueError(f'Native budget mismatch: expected {sorted(expected)}, actual {kept}')
                if job['backend_kind']=='vla-cache':
                    expected = n if cold or job['strategy']=='vanilla' else n-int(round(n*job['ratio']))
                    if kept != expected:
                        raise ValueError(f'Cache budget mismatch: expected {expected}, actual {kept}')
                pact_process = None
                if job['strategy']=='pact-vla':
                    raw_trace = getattr(loaded[-1], '_pact_raw_process_trace', None)
                    if raw_trace is None:
                        pact_process = dict(
                            selected_visual_indices=list(range(n)),
                            pact=None,
                            controller_called=False,
                        )
                    else:
                        keep_indices, score_info = raw_trace
                        image_start = int(score_info['image_token_start_index'])
                        image_end = image_start + int(score_info['image_token_length'])
                        visual_indices = keep_indices[(keep_indices >= image_start) & (keep_indices < image_end)]
                        pact_process = dict(
                            selected_visual_indices=(visual_indices - image_start).detach().cpu().tolist(),
                            pact=score_info.get('pact'),
                            controller_called=True,
                        )
                costs=prefill_cost(layer_visits) if collect_flops else {}
                if counter is not None:
                    total=counter.get_total_flops()
                    # A counter that misses inference_mode can count selection
                    # alone (nonzero!) while missing the entire LLM. Reject it.
                    if total < 2*costs['llm_prefill_eq9_T']*1e12:
                        raise ValueError('FLOP counter misses major LLM operations.')
                    flopfile.write(json.dumps(dict(call=len(calls)+1,**tracking,total_flops=total,
                        registered_operation_flops={str(k):v for k,v in counter.get_flop_counts()['Global'].items()},
                        note=METRIC_NOTES['FLOPs_profiled_call(T)']))+'\n')
                entry = dict(call=len(calls)+1, policy_ms=policy_ms, model_ms=sum(last_model_times),
                    cold=int(cold), visual_kept=kept, warm_sample=int(len(calls)>=job['warmup_calls'] and not profiled),
                    **tracking,profiled_flops=int(profiled),
                    **{k:v for k,v in costs.items() if k!='llm_layer_visit_counts'},
                    profiled_call_flops_T=counter.get_total_flops()/1e12 if counter is not None else None)
                calls.append(entry)
                writer.writerow(entry)
                audits.write(json.dumps(dict(call=len(calls), layer_lengths=layer_lengths,
                    visual_kept=kept, cache_selection=budget_state,flop_estimates=costs,
                    pact_process=pact_process, action=action.tolist(), **tracking), default=str)+'\n')
                if len(calls)%10 == 0 or len(calls)==1:
                    report()
                if job['mode']=='verify' and len(calls)>=job['verify_calls']:
                    raise VerificationComplete()
                return value
            except VerificationComplete:
                raise
            except Exception:
                policy_errors.append(traceback.format_exc())
                # Some upstream evaluators otherwise count inference errors as
                # ordinary task failures. Terminate instead of hiding bugs.
                raise SystemExit('Policy error; see result.json for traceback.')
        evaluation.get_action = get_action
        settings = {**job['settings'], 'local_log_dir':str(out/'internal_logs'),
                    'run_id_note':job['id']}
        if job['backend_kind']=='vla-cache':
            from pact_eval.checkpoints import shadow_checkpoint
            settings['pretrained_checkpoint'] = str(shadow_checkpoint(settings['pretrained_checkpoint'],out))
        sys.argv = [str(evaluator_path)]
        for key, value in settings.items():
            sys.argv += ['--'+key, str(value)]
        save(out/'invocation.json', {'argv':sys.argv, 'task_ids':selected})
        try:
            evaluation.eval_libero()
        except VerificationComplete:
            state = 'VERIFIED'
        else:
            state = 'COMPLETED'
        current = report()
        if state=='COMPLETED' and job['mode']=='eval' and current['episodes'] != job['expected_episodes']:
            raise ValueError(f"Episode count mismatch: {current['episodes']} != {job['expected_episodes']}")
        if state=='COMPLETED' and job['mode']=='verify':
            raise ValueError('Evaluator finished before the requested verification call count.')
        if current['episode_errors'] or policy_errors:
            raise ValueError('Backend reported an exception; this is not a valid benchmark.')
        if collect_flops and (current['flops_formula_calls']!=len(calls) or not current['flops_profiled_samples']):
            raise ValueError('FLOPs coverage incomplete.')
        if collect_flops and job['mode']=='eval' and set(current['flops_profiled_task_ids'])!=set(selected):
            raise ValueError('Some selected tasks lack FLOP profile samples.')
        if job['strategy']!='vanilla' and not adapter:
            n = 256 if job['model']=='openvla' else 512
            if not any(c['visual_kept'] < n for c in calls):
                raise ValueError('No pruning/reuse observed: method may be disabled or smoke is too short.')
        report(state)
        print('PACT_RESULT', json.dumps(result, default=str), flush=True)
    except BaseException as error:
        result['error'] = traceback.format_exc()
        report('FAILED')
        print(result['error'], flush=True)
        raise SystemExit(1) from error
    finally:
        for env in environments:
            try:
                env.close()
            except Exception as error:
                print('Environment cleanup warning:', error, flush=True)
        csvfile.close()
        audits.close()
        flopfile.close()


if __name__ == '__main__':
    main()
