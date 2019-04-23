I'm currently cleaning up the code, in the next few days the repository and all dependencies (especially nobos_commons and nobos_torch_lib) should be fully available, I'll then update the README with instructions etc.


# Installation
## Prerequisites
- Python 3.6+
- CUDA (tested with 9.0 and 10.0)
- CUDNN (tested with 7.5)
- PyTorch (tests with 1.0)
- OpenCV with Python bindings

A basic setup guide for Ubuntu 18.04 is available at: https://dennisnotes.com/note/20180528-ubuntu-18.04-machine-learning-setup/.
I set up my system like this, with the difference that I now use CUDA 10.0 and CUDNN 7.5, the blogpost will be updated sometime.

Note: The code runs on Windows, but there is somewhere a bug, so the whole thing runs on our system with only 10-30% of the FPS on Linux (Ubuntu 18.04).

## Setup
I use two of my libraries in this code, nobos_commons and nobos_torch_lib. These and their dependencies have to be installed first. In the following code example I assume a Python installation with virtualenvwrapper, if this is not used the code must be adapted accordingly.
A new virtual environment is created in the code, then PyTorch (with CUDA 10.0) is installed and then the required repositories cloned, dependencies installed and finally the required model weights loaded from our web server. The weights for YoloV3 and 2D Human Pose Recognition are originally from https://github.com/ayooshkathuria/pytorch-yolo-v3 and https://github.com/Microsoft/human-pose-estimation.pytorch. We have the weights on our server to ensure availability and version.

```bash
git clone https://github.com/noboevbo/nobos_commons.git
git clone https://github.com/noboevbo/nobos_torch_lib.git
git clone https://github.com/noboevbo/ehpi_action_recognition.git
mkvirtualenv ehpi_action_recognition -p python3 --system-site-packages
workon ehpi_action_recognition
pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install -r nobos_commons/requirements.txt 
pip install -r nobos_torch_lib/requirements.txt
pip install -r ehpi_action_recognition/requirements.txt  
pip install --upgrade nobos_commons
pip install --upgrade nobos_torch_lib
cd ehpi_action_recognition
sh get_models.sha

```
An example showing the whole pipeline on the webcam can be executed as follows:
```bash
export PYTHONPATH="~/path/to/ehpi_action_recognition:$PYTHONPATH"
python ehpi_action_recognition/run_ehpi.py
```
I haven't adapted the whole thing to the command line yet, changes can be made in the code. Examples for training and evaluation can be found in the files "train_ehpi.py" and "evaluate_ehpi.py".

# Reconstruct paper results
This repository contains code for our (submitted, as of 23.04.2019) publication on ITSC 2019 and ITS Journal Special Issue ITSC 2018. As the EHPI publication is not yet published and citable, we have used an LSTM approach for action recognition for the ITS Journal publication, which is based on the normalized EHPI inputs. We want to ensure that the results can be reproduced from our papers. Therefore, we provide our training and evaluation code in this repository. The results in our papers are reported as mean values from five training sessions with different seeds. As seeds we use 0, 104, 123, 142 and 200. We use fixed values so that the results are 100% reproducible, seeds 142 and 200 are randomly selected, 0 and 123 are seeds often used in other work and 104 is our office room number. 

## IEEE Intelligent Transportation Systems Conference (ITSC 2019)
- Our datasets are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_datasets.tar.gz
- Our trained models are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_models.tar.gz

Here is an example of the standard setup that should allow our training and evaluation code to be used directly:
```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_datasets.tar.gz
tar -xvf itsc_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_models.tar.gz
tar -xvf itsc_2019_models.tar.gz

```
Here is the direct link to the training code for the JHMDB dataset: [JHMDB Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/ehpi/train_ehpi_itsc_2019_jhmdb.py)<br/>
And here to the evaluation code: [JHMDB Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/ehpi/test_ehpi_itsc_2019_jhmdb.py)

Here is the direct link to the training code for the Use Case dataset: [Use Case Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/ehpi/train_ehpi_itsc_2019_ofp.py)</br>
And here to the evaluation code: [Use Case Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/ehpi/test_ehpi_itsc_2019_ofp.py)

## IEEE Transactions on Intelligent Transportation Systems - Special Issue 21st IEEE Intelligent Transportation Systems Conference (ITSC 2018)
- Our datasets are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_datasets.tar.gz
- Our trained models are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_lstm_models.tar.gz

Here is an example of the standard setup that should allow our training and evaluation code to be used directly:
```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_datasets.tar.gz
tar -xvf its_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_lstm_models.tar.gz
tar -xvf its_2019_lstm_models.tar.gz

```
Here is the direct link to the training code for the both datasets (ActionSim and Office): [ITS Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/lstm/train_its_journal_2019.py)<br/>
And here to the evaluation code: [ITS Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/lstm/test_its_journal_2019.py)

# Citation
Please cite the following papers if this code is helpful in your research.
Currently the publications to this repository are submitted, but not yet accepted or published. I will update the entries as soon as I have feedback about the submissions. A preprint for the ITSC 2019 publication is available [here](https://arxiv.org/abs/1904.09140) on arxiv.org.

```bash
@inproceedings{Ludl2019SimpleYE,
  title={Simple yet efficient real-time pose-based action recognition},
  author={Dennis Ludl and Thomas Gulde and Crist'obal Curio},
  year={2019}
}
```


# Open Source Acknowledgments
I used parts of the following open source projects in my code:

- YoloV3: https://github.com/ayooshkathuria/pytorch-yolo-v3
- 2D Human Pose Estimation: https://github.com/Microsoft/human-pose-estimation.pytorch
- Imbalanced dataset sampler: https://github.com/ufoym/imbalanced-dataset-sampler/

Thank you for making this code available!