.PHONY: default install train clean

default: train

install:
	python -m pip install -r requirements.txt

train:
	python train.py

clean:
	rm -rf mario_checkpoints/ mario_best_model/ mario_videos/ mario_logs/
