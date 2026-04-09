.PHONY: install install-gpu test data lgbm patchtst timesfm all clean

install:
	pip install -r requirements.txt

install-gpu:
	pip install -r requirements-gpu.txt

test:
	pytest tests/ -v

data:
	python -m src.pipeline --phase data

lgbm:
	python -m src.pipeline --phase lgbm

patchtst:
	python -m src.pipeline --phase patchtst

timesfm:
	python -m src.pipeline --phase timesfm

all:
	python -m src.pipeline --phase all

clean:
	rm -rf results/
