# Model-Based Deep Learning for Limited View PACT
## abstract
Photoacoustic Computed Tomography (PACT) is an emerging field in biomedical imaging used in e.g. breast cancer
detection. The imaging modality uses light to excite the object
under study, which then transforms this light into acoustic waves
via thermoelastic expansion. A well-known problem in PACT
is its reliance on big lasers combined with large ultrasonic
sensor arrays to acquire an image. Using more versatile and
less expensive options like a probe with a built-in light source
could contribute to making PACT a standard technique for
clinical applications. However, the signal acquired from a probe
is not sufficient for making a sufficiently accurate image. In
this paper, we propose a joint deep learning and compressed
sensing approach for finding the acoustic inverse. It achieves
this by finding a learned regularization for the ill-posed inverse
problem and then iteratively using this regularization to fill in
data acquired from artificial sensors surrounding the subject.
Results show that steps in the good direction are taken, but
more focus on the neural network is still needed to make it a
viable solution to the problem.

## Repository
This repository consists of the code that can be used to run and train the neural network. It contains the following: 

- Network: This folder contains the pieces of the network. It has a python implementation of the COLE and DAS algorithms as well as a python wrapper for KWave.
- Data: This package contains the dataloader
- Run.py: is used to run the network
- Trainer.py: is used to train the network
- Model-Based Deep Learning for Limited View PACT.pdf: Report for the project