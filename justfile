registry := "localhost:32000"
namespace := "ciroos-rag"

# List available recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

# Build all container images
build: build-arbiter build-hybrid build-graph

# Build arbiter image
build-arbiter:
    podman build -f Dockerfile.arbiter -t ragrouter/arbiter:latest .

# Build hybrid engine image
build-hybrid:
    podman build -f Dockerfile.hybrid -t ragrouter/hybrid-engine:latest .

# Build graph engine image
build-graph:
    podman build -f Dockerfile.graph -t ragrouter/graph-engine:latest .

# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

# Tag and push all images to MicroK8s registry
push: push-arbiter push-hybrid push-graph

# Push arbiter image
push-arbiter:
    podman tag ragrouter/arbiter:latest {{registry}}/ragrouter/arbiter:latest
    podman push --tls-verify=false {{registry}}/ragrouter/arbiter:latest

# Push hybrid engine image
push-hybrid:
    podman tag ragrouter/hybrid-engine:latest {{registry}}/ragrouter/hybrid-engine:latest
    podman push --tls-verify=false {{registry}}/ragrouter/hybrid-engine:latest

# Push graph engine image
push-graph:
    podman tag ragrouter/graph-engine:latest {{registry}}/ragrouter/graph-engine:latest
    podman push --tls-verify=false {{registry}}/ragrouter/graph-engine:latest

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

# Deploy all K8s manifests (namespace, secrets, all engines)
deploy: deploy-infra deploy-arbiter deploy-hybrid deploy-graph

# Create namespace and secrets
deploy-infra:
    #!/usr/bin/env bash
    set -euo pipefail
    kubectl apply -f k8s/namespace.yaml
    if ! kubectl get secret ragrouter-secrets -n {{namespace}} >/dev/null 2>&1; then
      echo "Creating ragrouter-secrets from litellm-secrets..."
      OPENAI_KEY=$(kubectl get secret litellm-secrets -n ciroos -o jsonpath='{.data.OPENAI_API_KEY}')
      kubectl create secret generic ragrouter-secrets -n {{namespace}} \
        --from-literal=openai-api-key="$(echo "$OPENAI_KEY" | base64 -d)"
    else
      echo "ragrouter-secrets already exists"
    fi

# Deploy arbiter
deploy-arbiter:
    kubectl apply -f k8s/arbiter.yaml

# Deploy hybrid engine
deploy-hybrid:
    kubectl apply -f k8s/hybrid-engine.yaml

# Deploy graph engine
deploy-graph:
    kubectl apply -f k8s/graph-engine.yaml

# ---------------------------------------------------------------------------
# Composite: build + push + deploy (per-service or all)
# ---------------------------------------------------------------------------

# Build, push, and deploy everything
all: build push deploy

# Build, push, and deploy arbiter only
all-arbiter: build-arbiter push-arbiter deploy-infra deploy-arbiter

# Build, push, and deploy hybrid engine only
all-hybrid: build-hybrid push-hybrid deploy-hybrid

# Build, push, and deploy graph engine only
all-graph: build-graph push-graph deploy-graph

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

# Show pod and service status
status:
    @echo "=== Pods ==="
    kubectl get pods -n {{namespace}} -o wide
    @echo ""
    @echo "=== Services ==="
    kubectl get svc -n {{namespace}}
    @echo ""
    @echo "=== Endpoints ==="
    kubectl get endpoints -n {{namespace}}

# Tail logs from all RAG pods
logs:
    kubectl logs -n {{namespace}} -l 'app in (arbiter, hybrid-engine, graph-engine)' -f --tail=100

# Tail logs from a specific service (arbiter, hybrid-engine, graph-engine)
logs-svc svc:
    kubectl logs -n {{namespace}} -l app={{svc}} -f --tail=200

# Restart all deployments (picks up new images with imagePullPolicy: Always)
restart: restart-arbiter restart-hybrid restart-graph

# Restart arbiter
restart-arbiter:
    kubectl rollout restart deployment/arbiter -n {{namespace}}

# Restart hybrid engine
restart-hybrid:
    kubectl rollout restart deployment/hybrid-engine -n {{namespace}}

# Restart graph engine
restart-graph:
    kubectl rollout restart deployment/graph-engine -n {{namespace}}

# Wait for all deployments to be ready
wait:
    kubectl rollout status deployment/arbiter -n {{namespace}} --timeout=120s
    kubectl rollout status deployment/hybrid-engine -n {{namespace}} --timeout=120s
    kubectl rollout status deployment/graph-engine -n {{namespace}} --timeout=120s

# Port-forward all services to localhost
port-forward:
    #!/usr/bin/env bash
    trap 'kill $(jobs -p) 2>/dev/null' EXIT
    kubectl port-forward -n {{namespace}} svc/arbiter 8000:8000 &
    kubectl port-forward -n {{namespace}} svc/hybrid-engine 8001:8001 &
    kubectl port-forward -n {{namespace}} svc/graph-engine 8004:8004 &
    echo "Forwarding:"
    echo "  arbiter        → localhost:8000"
    echo "  hybrid-engine  → localhost:8001"
    echo "  graph-engine   → localhost:8004"
    echo "Press Ctrl+C to stop"
    wait

# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

# Hit health endpoints on all services (requires port-forward or in-cluster)
health:
    #!/usr/bin/env bash
    set -e
    for svc in arbiter:8000 hybrid-engine:8001 graph-engine:8004; do
      name="${svc%%:*}"
      port="${svc##*:}"
      url="http://${name}.{{namespace}}.svc.cluster.local:${port}/health"
      echo -n "${name}: "
      kubectl exec -n {{namespace}} deploy/arbiter -- \
        python -c "import httpx; r=httpx.get('${url}', timeout=5); print(r.status_code, r.json())" \
        2>/dev/null || echo "UNREACHABLE"
    done

# Quick smoke test: send a query through the arbiter
smoke-test query="What is Ciroos?":
    #!/usr/bin/env bash
    kubectl exec -n {{namespace}} deploy/arbiter -- \
      python -c "
    import httpx, json
    r = httpx.post('http://localhost:8000/ask', json={
        'query': '{{query}}',
        'corpus_id': 'ciroos-docs',
        'chat_history': [],
        'constraints': {'max_latency_ms': 10000, 'must_cite': True},
    }, timeout=30)
    print(json.dumps(r.json(), indent=2))
    "

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run all unit tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov=ragrouter --cov-report=term-missing

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

# Delete all resources in ciroos-rag namespace
teardown:
    kubectl delete -f k8s/graph-engine.yaml --ignore-not-found
    kubectl delete -f k8s/hybrid-engine.yaml --ignore-not-found
    kubectl delete -f k8s/arbiter.yaml --ignore-not-found
    kubectl delete secret ragrouter-secrets -n {{namespace}} --ignore-not-found
    kubectl delete -f k8s/namespace.yaml --ignore-not-found
