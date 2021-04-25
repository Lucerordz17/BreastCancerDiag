pip3 -q install pydicom
pip -q uninstall -y kaggle
pip -q install --upgrade pip
pip3 -q install --upgrade kaggle

mkdir /root/.kaggle
echo '{"username":"lucerorodriguez","key":"f9960528042fead1f6039c3a876df225"}' > /root/.kaggle/kaggle.json

mkdir data
mkdir data/train
mkdir data/test
mkdir /content/logs

python3 BreastCancerDiag/dataset_prep/data.py