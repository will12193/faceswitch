#!/bin/sh
set -eux

# login() {
# 	[ -n "${CI_REGISTRY_PASSWORD:-}" ] &&
# 		docker login -u "${CI_REGISTRY_USER}" -p "$CI_REGISTRY_PASSWORD" "$CI_REGISTRY" &&
# 		return
# 	[ -z "${CI_REGISTRY_PASSWORD:-}" ] &&
# 		docker login -u "${CI_REGISTRY_USER}" "$CI_REGISTRY"
# }

build() {
	IMAGE="$1"
	DOCKERFILE="$2"
	CONTEXT="$3"
	STAGE="${4}"
	CACHE_FROM="${5}"

	# docker pull "$IMAGE:latest" || true
	docker build \
		--build-arg="CI_COMMIT_SHA=$CI_COMMIT_SHA" \
		--cache-from "$CACHE_FROM" \
		--progress=plain \
		-f "$DOCKERFILE" \
		--tag "$IMAGE:$CI_COMMIT_SHA" \
		--tag "$IMAGE:latest" \
		--target "$STAGE" \
        --platform linux/amd64 \
		"$CONTEXT"
}

# CI_REGISTRY="${CI_REGISTRY:-container.divisia.io}"
# CI_REGISTRY_USER="${CI_REGISTRY_USER:-${USER}}"
CI_COMMIT_SHA="${CI_COMMIT_SHA:-$(git rev-parse HEAD)}"

WHAT="${1:-container.will12193/faceswitch}"
DOCKERFILE="${2:-docker/Dockerfile}"
CONTEXT="${3:-.}"
STAGE="${4:-production}"
CACHE_FROM="${5:-$WHAT:latest}"

# login
build "$WHAT" "$DOCKERFILE" "$CONTEXT" "$STAGE" "$CACHE_FROM"