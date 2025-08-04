#!/bin/bash

docker build -f Dockerfile -t kraepelin-backend:latest .

docker tag kraepelin-backend:latest 081218068401/kraepelin-backend:latest

docker push 081218068401/kraepelin-backend:latest