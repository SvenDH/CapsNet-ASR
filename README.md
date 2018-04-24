# CapsNet-ASR
Phoneme recognition with Capsule Networks

Presentation 24 april link: https://docs.google.com/presentation/d/149ZALP9stKvSWqu2N12QfzwSR06X3CsfPLHTvgIRO0U/edit?usp=sharing

Extra ideas (possible individual research questions):

Since our frames are not necessarily of only one phone, we can maybe label them as 0.75A 0.25B and compare this with the classifier distribution. This would mean that (0.75A 0.25B) predicted as (0.6A 0.2B 0.2C) is better than (0.6A 0.1B 0.3C), even though in both predictions the amount of prediciton for A is the same.

Extra reading:

First paper to use 48 (39) instead of 61 labels: 
http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci 

Single recurrent neural network on TIMIT phoneme classification:
http://people.idsia.ch/~santiago/papers/IDSIA-04-08.pdf

Good general info about TIMIT and phoneme classification:
https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database

