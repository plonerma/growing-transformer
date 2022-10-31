# load dot-env variables
export $(egrep -v '^#' .env | xargs)

rsync -urltv -e ssh --exclude-from scripts/not_deployed . ${REMOTE_PATH}
