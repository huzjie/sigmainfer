# 部署

## Docker

```bash
docker build -t sigmainfer .
docker run -p 8000:8000 sigmainfer
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/
```

## Helm

```bash
helm install sigmainfer deploy/helm
```
