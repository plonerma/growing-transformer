# load dot-env variables
export $(egrep -v '^#' .env | xargs)


IGNORE_MODELS=(--exclude '*/pytorch_model.bin')

while getopts 'm' OPTION; do
  case "$OPTION" in
    m)
      echo "Downloading models as well.";
      IGNORE_MODELS=();
      ;;
    ?)
      echo "script usage: $0 [-m]" >&2
      exit 1
      ;;
  esac
done

shift "$(($OPTIND -1))"


echo Using options: ${IGNORE_MODELS[@]};


rsync -urltv ${IGNORE_MODELS[@]} -e ssh ${REMOTE_PATH}/outputs .
rsync -urltv ${IGNORE_MODELS[@]} -e ssh ${REMOTE_PATH}/multirun .
