.PHONY: start
start:
	uvicorn main:app --reload --port 8088

.PHONY: format
format:
	black .
	isort .