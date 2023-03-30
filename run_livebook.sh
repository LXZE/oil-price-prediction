docker run \
	-d \
	-it \
	-p 8080:8080 \
	-p 8081:8081 \
	--pull always \
	-u $(id -u):$(id -g) \
	-v $(pwd):/data \
	-e LIVEBOOK_PASSWORD="livebook_secret" \
	--name livebook \
	ghcr.io/livebook-dev/livebook
