VENV_PATH="venv"

if [ `readlink -e ${VENV_PATH}` ]; then
  echo "Virtualenv (${VENV_PATH}) exists. Delete first using:"
  echo "rm -r ${VENV_PATH}";
  exit 1
fi

echo "Creating virtualenv..."

python3 -m virtualenv ${VENV_PATH}

echo "Installing package"

. ${VENV_PATH}/bin/activate

pip install -e .

echo "Done"
