#######################
# BUILD DOCKER IMAGE  #
#######################

# Debug mode
set -x 


TIME=$(date  +%Y%m%d_%H%M)
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=apolloauto/apollo
fi


WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# Build image from APOLLO_ROOT, while use the specified Dockerfile.
docker build -t "${DOCKER_REPO}:dev-${TIME}" \
    -f "${WORKDIR}/docker/dev.dockerfile" \
    "${APOLLO_ROOT}"