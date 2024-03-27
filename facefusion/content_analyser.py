from typing import Any
from functools import lru_cache
from time import sleep
import cv2
import numpy
from tqdm import tqdm

import facefusion.globals
from facefusion import process_manager, wording
from facefusion.inference_pool import get_inference_session, clear_inference_session
from facefusion.typing import VisionFrame, ModelSet, Fps
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path
from facefusion.download import conditional_download

MODELS : ModelSet =\
{
	'open_nsfw':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
		'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
	}
}
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
	while process_manager.is_checking():
		sleep(0.5)
	model_path = MODELS.get('open_nsfw').get('path')
	return get_inference_session(model_path)


def clear_content_analyser() -> None:
	model_path = MODELS.get('open_nsfw').get('path')
	clear_inference_session(model_path)


def pre_check() -> bool:
	if not facefusion.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_url = MODELS.get('open_nsfw').get('url')
		process_manager.check()
		conditional_download(download_directory_path, [ model_url ])
		process_manager.end()
	return True


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
	global STREAM_COUNTER

	STREAM_COUNTER = STREAM_COUNTER + 1
	if STREAM_COUNTER % int(video_fps) == 0:
		return analyse_frame(vision_frame)
	return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
	content_analyser = get_content_analyser()
	vision_frame = prepare_frame(vision_frame)
	probability = content_analyser.run(None,
	{
		content_analyser.get_inputs()[0].name: vision_frame
	})[0][0][1]
	return probability > PROBABILITY_LIMIT


def prepare_frame(vision_frame : VisionFrame) -> VisionFrame:
	vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
	vision_frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	vision_frame = numpy.expand_dims(vision_frame, axis = 0)
	return vision_frame


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	frame = read_image(image_path)
	return analyse_frame(frame)


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	video_frame_total = count_video_frame_total(video_path)
	video_fps = detect_video_fps(video_path)
	frame_range = range(start_frame or 0, end_frame or video_frame_total)
	rate = 0.0
	counter = 0

	with tqdm(total = len(frame_range), desc = wording.get('analysing'), unit = 'frame', ascii = ' =', disable = facefusion.globals.log_level in [ 'warn', 'error' ]) as progress:
		for frame_number in frame_range:
			if frame_number % int(video_fps) == 0:
				frame = get_video_frame(video_path, frame_number)
				if analyse_frame(frame):
					counter += 1
			rate = counter * int(video_fps) / len(frame_range) * 100
			progress.update()
			progress.set_postfix(rate = rate)
	return rate > RATE_LIMIT
