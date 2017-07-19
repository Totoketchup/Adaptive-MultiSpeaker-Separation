# Build the GPU image
docker build -t das.gpu -f docker/Dockerfile.gpu .

# Launch the GPU image
nvidia-docker run -it -p 8888:8888 -p 6006:6006 das.gpu
