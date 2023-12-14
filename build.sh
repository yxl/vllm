#!/usr/bin/env bash

DOCKER_BRIDGE_SUBNET=192.162.0.0/16

# 部署脚本出错，自动清理 docker
DID_CLEAN_UP=0
# the cleanup function will be the exit point
cleanup () {
  if [ "$DID_CLEAN_UP" -eq 1 ]; then
    return 0;
  fi
  echo "清理中..."
  docker-compose down
  DID_CLEAN_UP=1
}
trap cleanup ERR INT TERM EXIT

echo "build vllm镜像"
docker-compose build || { echo "构建镜像失败"; exit 1; }

echo "部署vllm服务"
docker network inspect sagegpt_network >/dev/null 2>&1 || \
    docker network create --driver=bridge --subnet="$DOCKER_BRIDGE_SUBNET" sagegpt_network || { echo "创建虚拟子网失败"; exit 1; }
docker-compose up -d || { echo "部署失败"; exit 1; }

DID_CLEAN_UP=1