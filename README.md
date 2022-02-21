# SelfDriving-RL-Torcs
## Intro
Applicazione di tecniche di Reinforcement Learning (Q-Learning e Deep Q-Network) per insegnare ad un'auto a guidare autonomamente utilizzando il simulatore TORCS.

## Installazione
La versione usata di Torcs è la  1.3.4 disponibile al link:
https://sourceforge.net/projects/torcs/files/torcs-win32-bin/1.3.4/ 
<br>
La SCR patch utilizzata è disponibile al link:
https://sourceforge.net/projects/cig/files/SCR\%20Championship/Server\%20Linux/2.1/
<br>
Clona la repository
```sh
git clone https://github.com/ScaramuzzinoGiovanna/SelfDriving-RL-Torcs.git
```
## Procedura per l'installazione di TORCS:
```sh
./configure
make
sudo make install
make datainstall

sudo apt install xautomation
sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev
sudo apt-get install libvorbis-dev
```
** se il make dà un errore, è possibile risolverlo modificando i files:
- geometry.cpp: mettendo std:: davanti ad isnan  
- OpenALMusicPlayer.cpp: nullptr al posto di ‘\0’

# Requisiti
Per eseguire il codice sono necessari:
- __Python 3.6__
- __tensorflow 2.0__, __Keras__ 
- __Matplotlib__, __Numpy__, __Matplotlib__
