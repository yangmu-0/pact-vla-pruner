"""Pure-stdlib planning: never imports a model or starts a GPU job."""
import ast
import json
import os
from pathlib import Path

MODELS = ('openvla', 'oft')
STRATEGIES = ('vanilla', 'fastv', 'sparsevlm', 'divprune', 'vla-cache', 'vla-pruner')
SUITES = {'spatial': ('libero_spatial', 'spatial'), 'object': ('libero_object', 'object'),
          'goal': ('libero_goal', 'goal'), 'long': ('libero_10', '10')}

def expand(values, choices):
    values = [v.lower().replace('_', '-') for v in values]
    if 'all' in values:
        if len(values) != 1:
            raise ValueError('Use all by itself, not alongside individual choices.')
        return list(choices)
    for value in values:
        if value not in choices:
            raise ValueError(f'Unsupported choice {value}; choose from {list(choices)}')
    return list(dict.fromkeys(values))

def suite_keys(values):
    aliases = {v[0]: k for k,v in SUITES.items()}
    aliases.update(libero_long='long', libero_10='long')
    return expand([aliases.get(v, v) for v in values], SUITES)

def ratio(value):
    value = str(value).strip()
    percent = value.endswith('%')
    number = float(value.rstrip('%'))
    if percent or number >= 1:
        number /= 100
    if not 0 < number < 1:
        raise ValueError('Ratios must be strictly between 0 and 1 (or 0% and 100%). Use vanilla for zero.')
    return number

def fields_from_file(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GenerateConfig')
    return {n.target.id for n in config.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}

def backend(root, family, kind):
    config = json.loads((root/'configs/backends.json').read_text())
    item = config['backends'][('cache' if kind == 'vla-cache' else 'native')+'_'+family]
    cwd = root/item['cwd']
    python = Path(os.environ.get('PACT_CONDA_ROOT', config['conda_root']))/'envs'/item['environment']/'bin/python'
    search = [root/item['transformers']] if item['transformers'] else []
    search += [cwd, root/'dependencies/LIBERO', root]
    return {'cwd': str(cwd), 'python': str(python), 'environment': item['environment'],
            'pythonpath': os.pathsep.join(map(str, search)), 'transformers_override': item['transformers']}

def baseline_kind(strategy):
    return strategy if strategy in ('divprune', 'vla-cache') else 'native'

def make_jobs(root, args):
    models = expand(args.model, MODELS)
    strategies = expand(args.strategy, STRATEGIES)
    suites = suite_keys(args.suite)
    ratios = list(dict.fromkeys(ratio(r) for r in args.ratio))
    tasks = list(dict.fromkeys(args.task_ids)) if args.task_ids is not None else list(range(10))
    if not tasks or any(i < 0 or i > 9 for i in tasks):
        raise ValueError('--task-ids must be in 0..9 for these LIBERO suites.')
    if not 1 <= args.trials <= 50:
        raise ValueError('--trials must be 1..50 (available standard init-state budget).')
    if not 0 <= args.seed < 2**32:
        raise ValueError('--seed must be an unsigned 32-bit integer.')
    if not 1 <= args.prune_layer <= 31:
        raise ValueError('--prune-layer must be 1..31 (selection requires a preceding attention layer).')
    if args.checkpoint and (len(models) != 1 or len(suites) != 1):
        raise ValueError('A checkpoint override requires exactly one model and one suite.')
    jobs, seen = [], set()
    methods = []
    for strategy in strategies:
        kind = args.baseline_backend if strategy == 'vanilla' else baseline_kind(strategy)
        if args.with_baseline and strategy != 'vanilla':
            methods.append(('vanilla', baseline_kind(strategy), 0.0))
        methods += [(strategy, kind, r) for r in ([0.0] if strategy == 'vanilla' else ratios)]
    for family in models:
        for suite_key in suites:
            suite, suffix = SUITES[suite_key]
            for strategy, kind, pruning in methods:
                key = (family, suite, strategy, kind, pruning)
                if key in seen:
                    continue
                seen.add(key)
                if family == 'openvla' and kind == 'divprune' and args.decode_cache == 'on':
                    raise ValueError('The validated DivPrune OpenVLA adapter requires --decode-cache off/auto.')
                if kind == 'vla-cache' and args.decode_cache == 'off':
                    raise ValueError('VLA-Cache uses its official cached backend; --decode-cache off is incompatible.')
                spec = backend(root, family, kind)
                cwd = Path(spec['cwd'])
                checkpoint = str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else str(
                    cwd/'checkpoints'/('openvla-7b-'+('oft-' if family == 'oft' else '')+'finetuned-libero-'+suffix))
                settings = dict(pretrained_checkpoint=checkpoint, task_suite_name=suite, center_crop=True,
                                num_trials_per_task=args.trials, num_steps_wait=10, seed=args.seed,
                                use_wandb=False, save_rollout_videos=args.save_video)
                if kind == 'vla-cache':
                    settings['use_vla_cache'] = strategy == 'vla-cache'
                else:
                    settings.update(use_fastv=(strategy in ('fastv', 'sparsevlm', 'vla-pruner') if family == 'openvla' else strategy == 'fastv'),
                        sparsevlm=strategy == 'sparsevlm', use_prefil_attention=args.prefill_attention,
                        fastv_k=args.prune_layer, fastv_r=pruning)
                    if family == 'openvla':
                        settings.update(use_cache=args.decode_cache == 'on', use_temporal=strategy == 'vla-pruner',
                                        use_text_vision_selection=strategy == 'sparsevlm', temporal_w=3, temporal_gamma=.8)
                    else:
                        settings.update(use_vla_cache=False, use_vla_pruner=strategy == 'vla-pruner',
                                        use_pact_vla=False, fastv_attention_source='prefill',
                                        vla_pruner_layer=15, vla_pruner_mode='semantic_action',
                                        merge_local_lora_adapter=False)
                if family == 'oft':
                    settings.update(num_images_in_input=2, use_proprio=True, num_open_loop_steps=8,
                                    use_l1_regression=True, use_diffusion=False, use_film=False,
                                    lora_rank=32, env_img_res=256)
                for name, value in args.backend_option:
                    name = name.lstrip('-').replace('-', '_')
                    protected = {'pretrained_checkpoint','task_suite_name','num_trials_per_task','seed','local_log_dir',
                                 'run_id_note','use_fastv','sparsevlm','use_vla_pruner','use_vla_cache','use_pact_vla','fastv_r'}
                    if name in protected:
                        raise ValueError(f'{name} is controlled by the unified CLI; do not override its strategy/denominator.')
                    settings[name] = value
                budget_label = f'{pruning*100:g}'
                identifier = f'{family}_{suite_key}_{strategy}_{kind}_{budget_label}'
                jobs.append(dict(id=identifier, model=family, strategy=strategy, backend_kind=kind,
                    suite=suite, suite_key=suite_key, ratio=pruning,
                    ratio_semantics='target_final_reuse' if strategy == 'vla-cache' else 'visual_pruning',
                    task_ids=tasks, trials=args.trials, expected_episodes=len(tasks)*args.trials,
                    mode=args.mode, verify_calls=args.verify_calls, warmup_calls=args.warmup_calls,
                    collect_flops=args.collect_flops, flops_sample_interval=args.flops_sample_interval,
                    seed=args.seed, state='PLANNED', settings=settings, **spec))
    if args.condition_order=='method':
        jobs.sort(key=lambda j:(j['strategy']!='vanilla',STRATEGIES.index(j['strategy']),j['ratio'],
                               j['backend_kind'],list(SUITES).index(j['suite_key']),MODELS.index(j['model'])))
    return jobs

def validate_job(root, job):
    cwd = Path(job['cwd'])
    evaluator = cwd/'experiments/robot/libero/run_libero_eval.py'
    if not Path(job['python']).is_file() or not evaluator.is_file():
        raise ValueError(f"Backend or Python environment missing: {job['id']}")
    allowed = fields_from_file(evaluator)
    unknown = set(job['settings']) - allowed
    # Old cache evaluators have no save-video setting. Their default video function
    # is suppressed process-locally by worker.py when --save-video was not selected.
    if unknown == {'save_rollout_videos'}:
        job['settings'].pop('save_rollout_videos')
        unknown.clear()
    if unknown:
        raise ValueError(f"Unsupported backend options for {job['id']}: {sorted(unknown)}")
    cp = Path(job['settings']['pretrained_checkpoint'])
    for name in ('config.json','dataset_statistics.json'):
        if not (cp/name).is_file():
            raise ValueError(f'Missing checkpoint file: {cp/name}')
    index = cp/'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for shard in shards:
        if not (cp/shard).is_file():
            raise ValueError(f'Missing/broken checkpoint shard: {cp/shard}')
    if job['model']=='oft':
        for prefix in ('action_head', 'proprio_projector'):
            if not any(p.is_file() for p in cp.glob(prefix+'*.pt')):
                raise ValueError(f'Missing OFT auxiliary checkpoint: {cp}/{prefix}*.pt')

def environment(root, job, gpu):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('DIVPRUNE_', 'VLACACHE_', 'PACT_EVAL_')):
            env.pop(key)
    env.update(PYTHONPATH=job['pythonpath'], CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OPENVLA_SKIP_CHECKPOINT_SYNC='1',
        OPENVLA_PRUNING_REPORT_ONCE='1', TF_CPP_MIN_LOG_LEVEL='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return env
