format:
	black .

lint:
	npx pyright .

all: format lint
