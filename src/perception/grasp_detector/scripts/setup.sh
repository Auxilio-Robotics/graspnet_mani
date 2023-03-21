
# set path to grasp detector. go to configs and install requirements.txt
cd $(pwd)/config/
pip install -r requirements.txt
cd ..

cd pointnet2
python setup.py install

cd ../knn
python setup.py install

cd ../graspnetAPI
pip install .



