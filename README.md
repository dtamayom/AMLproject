# AMLproject
Advanced Machine Learning Project Proposal. 
Catalina Botía, Isabella Ramos, Daniela Tamayo. 

Universidad de los Andes, 2020

## Getting Started
Clone this repository.
Create a new environment following these commands: 
```
$ conda create -n opensim -c kidzik -c conda-forge opensim python=3.6.2
$ conda activate opensim
$ conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
$ pip install git+https://github.com/stanfordnmbl/osim-rl.git@ver2.1
$ pip install matplotlib
```
## Environment
The environment used is taken from the [NeurIPS 2018 Challenge: Learning to Run](https://github.com/stanfordnmbl/osim-rl) which is modified with a prosthetic leg for the purpose of this project.

## Running the code
Make sure to check the arguments.py file. 
### To train
```
$ CUDA_VISIBLE_DEVICES=0 python main.py 
```
### To evaluate
```
$ python Test.py
```
The final trained model for best and last rewards can be found [in this link](https://drive.google.com/drive/folders/1YtAh_Zt_aVgzBeNTtUcbZae2gE5aaGsX?usp=sharing) or in BCV002 /media/user_home0/dtamayo/AMLproject/saved_model_final. Download it and make sure to have it saved on a folder with the same name as the `checkpoint_dir` parameter in the `arguments.py` file. Run on MobaXTerm to visualize the skeleton!
