from typing import Optional, Dict
from onnxruntime import InferenceSession

import facefusion
import facefusion.globals
from facefusion.execution import apply_execution_provider_options
from facefusion.typing import InferencePool

INFERENCE_POOL : Optional[InferencePool] = {}
DEVICE_USAGE : Dict[str, int] = {}


def get_inference_session(model_path : str) -> InferenceSession:
	device_id = min(facefusion.globals.execution_devices, key = lambda device_usage: DEVICE_USAGE.get(device_usage, 0))

	if model_path not in INFERENCE_POOL:
		INFERENCE_POOL[model_path] = {}
	if device_id not in INFERENCE_POOL[model_path] or INFERENCE_POOL[model_path][device_id] is None:
		INFERENCE_POOL[model_path][device_id] = InferenceSession(model_path, providers = apply_execution_provider_options(facefusion.globals.execution_providers, device_id))
	DEVICE_USAGE[device_id] = DEVICE_USAGE.get(device_id, 0) + 1
	return INFERENCE_POOL[model_path][device_id]


def clear_inference_session(model_path : str) -> None:
	if model_path in INFERENCE_POOL:
		INFERENCE_POOL[model_path] = {}
