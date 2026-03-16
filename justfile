registry := "localhost:32000"
namespace := "ciroos-rag"

# List available recipes
default:
    @just --list

# Build both container images
build:
    podman build -f Dockerfile.arbiter -t ragrouter/arbiter:latest .
    podman build -f Dockerfile.hybrid -t ragrouter/hybrid-engine:latest .

# Tag and push images to MicroK8s registry
push:
    podman tag ragrouter/arbiter:latest {{registry}}/ragrouter/arbiter:latest
    podman tag ragrouter/hybrid-engine:latest {{registry}}/ragrouter/hybrid-engine:latest
    podman push --tls-verify=false {{registry}}/ragrouter/arbiter:latest
    podman push --tls-verify=false {{registry}}/ragrouter/hybrid-engine:latest

# Deploy K8s manifests (namespace, secrets, services)
deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl apply -f k8s/namespace.yaml
    # Copy OpenAI API key from litellm-secrets in ciroos namespace
    if ! kubectl get secret ragrouter-secrets -n {{namespace}} >/dev/null 2>&1; then
      echo "Creating ragrouter-secrets from litellm-secrets..."
      OPENAI_KEY=$(kubectl get secret litellm-secrets -n ciroos -o jsonpath='{.data.OPENAI_API_KEY}')
      kubectl create secret generic ragrouter-secrets -n {{namespace}} \
        --from-literal=openai-api-key="$(echo "$OPENAI_KEY" | base64 -d)"
    else
      echo "ragrouter-secrets already exists"
    fi
    kubectl apply -f k8s/arbiter.yaml
    kubectl apply -f k8s/hybrid-engine.yaml

# Build, push, and deploy
all: build push deploy

# Tail logs from both pods
logs:
    kubectl logs -n {{namespace}} -l 'app in (arbiter, hybrid-engine)' -f --tail=100

# Show pod status
status:
    kubectl get pods -n {{namespace}} -o wide
    @echo "---"
    kubectl get svc -n {{namespace}}

# Restart deployments (force image pull)
restart:
    kubectl rollout restart deployment/arbiter -n {{namespace}}
    kubectl rollout restart deployment/hybrid-engine -n {{namespace}}

# Port-forward arbiter (8000) and hybrid-engine (8001)
port-forward:
    #!/usr/bin/env bash
    kubectl port-forward -n {{namespace}} svc/arbiter 8000:8000 &
    kubectl port-forward -n {{namespace}} svc/hybrid-engine 8001:8001 &
    echo "Forwarding arbiter → localhost:8000, hybrid-engine → localhost:8001"
    echo "Press Ctrl+C to stop"
    wait

# Delete all resources
teardown:
    kubectl delete -f k8s/hybrid-engine.yaml --ignore-not-found
    kubectl delete -f k8s/arbiter.yaml --ignore-not-found
    kubectl delete -f k8s/namespace.yaml --ignore-not-found
