# Convenience targets for local dev. Activate the conda env first:
#   conda activate privacy-filter

.PHONY: install dev test test-fast test-cov test-real run docker-build docker-run clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest

test-fast:
	pytest -x -q

test-cov:
	pytest --cov=app --cov-report=term-missing --cov-report=html

# Smoke-test against the real openai/privacy-filter checkpoint (~1.5 GB download)
test-real:
	RUN_REAL_MODEL_SMOKE=1 pytest -m requires_model -v

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

docker-build:
	docker build -t privacy-filter .

docker-run:
	docker run --rm -p 8080:8080 -v "$(PWD)/data:/tmp/data" privacy-filter

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
