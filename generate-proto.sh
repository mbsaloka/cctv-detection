#! /bin/bash

# Client
protoc -I=./proto cctv.proto --js_out=import_style=commonjs:./proto --grpc-web_out=import_style=commonjs,mode=grpcwebtext:./proto

# Server
python3 -m grpc_tools.protoc -I./proto --python_out=./server --grpc_python_out=./server ./proto/cctv.proto
