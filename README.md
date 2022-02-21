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
Clona la repository:
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
- __Matplotlib__, __Numpy__

# Files Progetto
Il progetto è stato suddiviso per praticità in due cartelle, essendo due gli esperimenti effettuati:
- __Q-learning__: contenente il codice riguardante la realizzazione dell'algoritmo di Q-learning applicato al nostro caso di studio.
- __Deep Q-learning__: contenente il codice riguardante la realizzazione dell'algoritmo di Deep Q-learning applicato al nostro caso di studio.


I file in comune sono:

 - environment.py: file che prevede la realizzazione della classe Environment;
 - snakeoil3_gym.py: per la connessione client-server;
 - config_practice_race.py: per la selezione del circuito e della posizione di start dell'auto;
 - autostart.sh: per l'apertura in modo automatico di torcs in modo da poter visualizzare la simulazione
 - utility.py: contenente funzioni di utility;
   <br>
   
Nella cartella Q-learning gli altri files usati sono:

  - discretization.py: per la discretizzazione dello spazio degli stati
  - q_table.py: per la realizzazione della q-table
  - q_learning_train.py: per l'implementazione dell'algoritmo di apprendimento (file di main per l'addestramento)
  - q_learning_test.py: per l'implementazione del test per la valutazione di quanto appreso
  - plots.py: per la realizzazione dei plots
   
Mentre per la Deep-Q-Learning:

  - replay_buffer.py: implementazione del replay buffer
  - net.py: implementazione della rete neurale
  - agent.py: dqn agent che usa il replay buffer e costruisce ed aggiorna le reti neurali
  - dqn_train: implementazione dell'algoritmo di apprendimento (file di main per l'addestramento)
  - dqn_test.py: implementazione del test per la valutazione di quanto appreso
