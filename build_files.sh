mkdir -p static
export STATIC_ROOT=static
pip3 install -r requirements.txt
python3 manage.py collectstatic --noinput