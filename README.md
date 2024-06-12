# [Deep Learning] - 2023/2024

### Authors:
- Iván Hernández, [[iv97n_solos]](https://github.com/iv97n)
- Marcel Fernández, [[marcel_drawer]](https://github.com/u198734)
- Alejandro Pastor, [[alejandro_el_diablo]](https://github.com/u199327)

Dataset: [COCO Dataset, Image Captioning 2014](https://cocodataset.org/#home)  

### TL;DR  
This repository contains code for the execution of a local image caption generator, which has been designed based on the architecture described in the paper [
Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). 

_IMPORTANT NOTE_ : Only the main.py script is expected to be functional under the current project settings,
nevertheless, other scripts like evaluate.py and train.py have also been utilized during the development of the project and have also played a fundamental role.  

### Presequisites
- [Python Poetry](https://python-poetry.org/docs/)
### Minimal setup
- Clone the repository into your local machine
```bash
git clone git@github.com:iv97n/image-captioning.git
```
- In the root directory of the project create a folder called _model_ for storing the pretrained model .ckpt file
```bash
cd image-captioning
mkdir model
```
- Download the [_nic.ckpt_](https://drive.google.com/file/d/1aDKoQWUp-YmMxA-tRhdd2XKF02YXGrcs/view?usp=drive_link) file containing the model weights and save it into the _model_ directory you just created
- Also in the root directory create a folder called _data_ for storing the images you would like to pass to the model
```bash
mkdir data
```
- Install the dependecies running poetry install in the project root directory. This might take a few minutes.
```bash
poetry install
```
- Now you are ready to rock some captions 😎

### Caption generation
- Make sure the image you want to caption is within  the _data_ folder
- Run the following command, substituting _example.jpg_ with your image name
```bash
poetry run python main.py --imagepath data/example.jpg
```
