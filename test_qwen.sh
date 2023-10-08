#!/usr/bin/env bash
docker compose build qwen && docker compose run qwen python test.py
