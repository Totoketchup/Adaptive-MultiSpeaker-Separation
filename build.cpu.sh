# Build the CPU image
docker build -t das.cpu -f docker/Dockerfile.cpu .

# Launch the CPU image
docker run -it -p 8888:8888 -p 6006:6006 das.cpu  
