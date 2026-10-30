"""Actionable, field-addressed validation feedback for the local workbench."""
from __future__ import annotations

import math
import re


FIELD_GROUPS = {
    **dict.fromkeys(('checkpoint', 'completion_checkpoint', 'model', 'ablation', 'num_joints', 'num_out_points', 'evaluation_weight_id'), 'model'),
    **dict.fromkeys(('data_root', 'split_strategy', 'seed', 'num_points', 'input_dim', 'seq_len', 'partition', 'split_chunk_size', 'max_timestamp_gap_ms', 'min_valid_points', 'pcc_label_radius'), 'data'),
    **dict.fromkeys(('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer'), 'training'),
    **dict.fromkeys(('scheduler_name', 'scheduler_eta_min', 'scheduler_step_size', 'scheduler_gamma'), 'schedule'),
    **dict.fromkeys(('device', 'gpu_ids', 'num_workers', 'torch_threads', 'cache_size', 'environment', 'limit_batches', 'allow_partial'), 'runtime'),
    **dict.fromkeys(('output_dir', 'visual', 'save_predictions'), 'output'),
}


def exception_issue(exc, *, name='configuration', group=None):
    """Preserve structured selection errors and locate failures from lower layers."""
    structured = getattr(exc, 'issue', None)
    if isinstance(structured, dict):
        return dict(structured)
    message = str(exc).strip()
    lower = message.lower()
    if name in ('configuration', 'split_strategy'):
        field = re.match(r'^([a-z_]+)(?:\s|:)', message)
        if field and field.group(1) in FIELD_GROUPS:
            name = field.group(1)
        elif 'partition requires' in lower:
            name = 'partition'
    if name == 'configuration':
        if 'gpu' in lower:
            name = 'gpu_ids'
        elif 'cuda' in lower:
            name = 'device'
        elif isinstance(exc, (ImportError, ModuleNotFoundError)):
            name = 'environment'
        elif 'checkpoint' in lower or 'state_dict' in lower:
            name = 'checkpoint'
    return dict(name=name, group=group or FIELD_GROUPS.get(name, 'model'), severity='error',
                message=message or f'Validation could not complete ({type(exc).__name__}).',
                details=f'{type(exc).__name__}: {message or "No additional error details were provided."}')


def safe_value(value, redact=lambda value: value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_value(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(item, redact) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def enrich_issue(issue, *, config=None, redact=lambda value: value):
    """Every warning/error remains useful even if its field is hidden in the UI."""
    result = dict(issue)
    name = result.get('name') or 'configuration'
    message = str(result.get('message') or '').strip()
    if not message:
        message = 'The dataset manifest has no valid sequence inventory.' if name == 'data_root' else 'Validation could not complete for this setting.'
    if name == 'output_dir' and 'checkpoint' in message.lower():
        name = 'checkpoint'
        result['group'] = 'model'
    result.update(name=name, group=result.get('group') or FIELD_GROUPS.get(name, 'model'),
                  severity='warning' if result.get('severity') == 'warning' else 'error', message=message)
    lower = message.lower()
    code, suggestion = 'parameter_invalid', 'Correct the reported value and retry validation. Keep locked evaluation parameters consistent with the selected training weight.'
    if name in ('checkpoint', 'completion_checkpoint', 'evaluation_weight_id'):
        code, suggestion = 'checkpoint_invalid', 'Refresh the weight list and select an existing compatible training checkpoint. Restore a missing or damaged official checkpoint from the official release; keep its recorded architecture and input settings.'
        if any(text in lower for text in ('size mismatch', 'missing key', 'unexpected key', 'state_dict')):
            code, suggestion = 'checkpoint_incompatible', 'Select the checkpoint that matches this model, variant, input features and output joints. Refresh the selected weight settings instead of editing locked architecture parameters.'
    elif name in ('gpu_ids', 'device'):
        code, suggestion = 'gpu_selection_invalid', 'Refresh the GPU list and select one available GPU for evaluation, or switch Device to CPU. Training may select multiple supported GPUs.'
        if 'cuda is unavailable' in lower:
            code, suggestion = 'cuda_unavailable', 'Switch Device to CPU, or restart the service with the repository CUDA environment after installing the matching CUDA profile.'
    elif name == 'environment':
        code, suggestion = 'environment_unavailable', 'Start the service with the repository Python environment and its required compute profile. Resolve the import or initialization error shown in Error details, then retry.'
    elif name == 'data_root':
        code, suggestion = 'dataset_unavailable', 'Open Dataset and download or repair the dataset. Verify that the service data directory contains a valid manifest.json and its referenced sequence files, then retry.'
    elif name in ('split_strategy', 'seq_len', 'seed', 'partition', 'split_chunk_size'):
        code, suggestion = 'dataset_split_invalid', 'Verify the installed dataset and the recorded split/cache metadata. For evaluation, select a compatible training weight or restore its recorded settings; do not change the locked split to bypass this error.'
        if '--trust-pickle' in message:
            code, suggestion = 'dataset_trust_required', 'For a dataset from a trusted source, restart the service with --trust-pickle so its recorded data cache can be prepared, then retry.'
    elif name == 'output_dir':
        code, suggestion = 'output_unavailable', 'Refresh the automatically assigned log directory and use an unused run/test ID. Wait for any task that owns this output to finish; existing logs are not overwritten.'
    elif name == 'allow_partial':
        code, suggestion = 'dataset_partial', 'Use Dataset to download or repair all required packages. Official weight evaluation requires the complete recorded dataset and split.'
    elif name == 'limit_batches':
        code, suggestion = 'batch_limit_incomplete', 'Remove the batch limit for a complete result. If a limit is intentionally used for training diagnostics, its output cannot represent a full evaluation.'
    elif name == 'batch_size' and 'batchnorm' in lower:
        code, suggestion = 'batch_size_singleton', 'Choose a batch size for which the training sample count does not leave a final batch of one sample.'
    elif name in ('visual', 'save_predictions'):
        code, suggestion = 'output_option_invalid', 'Enable prediction PKL visualization only for a pose Test task. Use the output options supported by the selected task.'
    elif name == 'configuration':
        code, suggestion = 'validation_failed', 'Review Error details and refresh the selected training weight settings. If the same internal error persists, provide these details with the model and task so the validation failure can be corrected.'
    elif name == 'learning_rate' and result['severity'] == 'warning':
        code, suggestion = 'learning_rate_high', 'Reduce Learning rate to the experiment recommendation (normally 0.0005), or explicitly review the intended high-rate experiment before retrying.'
    elif name == 'epochs' and result['severity'] == 'warning':
        code, suggestion = 'epochs_high', 'Reduce Epochs to the planned training budget, or review the unusually long experiment before retrying.'
    numeric_constraints = {
        'num_workers': 'an integer greater than or equal to 0', 'seed': 'an integer greater than or equal to 0',
        'batch_size': 'an integer greater than or equal to 1 (use at least 2 for training with BatchNorm)',
        **dict.fromkeys(('num_points', 'seq_len', 'num_joints', 'num_out_points', 'epochs', 'torch_threads', 'scheduler_step_size', 'cache_size', 'split_chunk_size'), 'an integer greater than or equal to 1'),
        **dict.fromkeys(('learning_rate', 'pcc_label_radius', 'max_timestamp_gap_ms'), 'a finite number greater than 0'),
        **dict.fromkeys(('weight_decay', 'scheduler_eta_min'), 'a finite number greater than or equal to 0'),
        'input_dim': '3 (XYZ), 4 (XYZ + Doppler), or 5 (XYZ + Doppler + SNR)',
        'partition': 'three nonnegative finite ratios with a positive sum, for example 0.8 / 0.1 / 0.1',
        'scheduler_gamma': 'a finite number greater than 0 and at most 1',
        'min_valid_points': 'an integer greater than or equal to 0',
    }
    numeric_failure = code == 'parameter_invalid' or code == 'dataset_split_invalid' and any(text in lower for text in ('must be', 'invalid value', 'nonnegative', 'positive sum'))
    if result['severity'] == 'error' and name in numeric_constraints and numeric_failure:
        expected = numeric_constraints[name]
        result.setdefault('expected', expected)
        suggestion = f'Set {name.replace("_", " ")} to {expected}. For a locked evaluation setting, restore the selected weight settings or select a compatible training record.'
    if message == 'Unsupported configuration parameter.':
        code, suggestion = 'parameter_unsupported', 'Reload the page and refresh the selected weight metadata. Remove this unsupported parameter from external/custom configuration requests.'
    result.setdefault('code', code)
    result.setdefault('suggestion', suggestion)
    if config is not None and name in config and 'actual' not in result:
        result['actual'] = config[name]
    return safe_value(result, redact)
