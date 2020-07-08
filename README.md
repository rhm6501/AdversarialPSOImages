# AdversarialPSO
This repository contains code to reproduce the AdversarialPSO attack results as reported in "[They Might NOT Be Giants: Crafting Black-Box Adversarial Examples with Fewer Queries Using Particle Swarm Optimization](https://arxiv.org/abs/1909.07490)". 

## Getting Started
The run this code, first install the required dependencies by running:
```
pip install -r requirements.txt
``` 

For the ImageNet test set, you can download the complete test set [here](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz). To launch the attack, download the img.tar.gz file, uncompress it, and place the images in the imagenet/images directory.

To launch the attack on 224x224 imagenet images, download the [x_val_0_10000.npy](https://www.dropbox.com/s/v3qvoi2mbtmh259/x_val_0_10000.npy?dl=0) file and place it in the imagenet folder. For all Imagenet attacks, the -d argument can be used to specify both imagenet299 and imagenet224.

After running the attack, the results will be placed in the Results directory.

## Reproducing the Results 
To reproduce the AdversarialPSO results reported in the paper, execute the following:
### CIFAR-10
For the untargeted AdversarialPSO attack, execute:
```
python AdversarialPSO.py
```
For the targeted AdversarialPSO attack, execute:
```
python AdversarialPSO.py -t
```
### MNIST
For the untargeted AdversarialPSO attack, execute:
```
python AdversarialPSO.py -d MNIST --blockSize 2 --maxChange 0.3
```

For the targeted AdversarialPSO attack, execute:
```
python AdversarialPSO.py -d MNIST --blockSize 2 --maxChange 0.3 -t
```

### Imagenet
For the untargeted AdversarialPSO attack on the indices provided by the Parsimonious attack, execute:
```
python AdversarialPSO.py -d imagenet299 --maxChange 0.1 --blockSize 32 --pars
```

Note: maxChange was set to 0.1 because the Keras implementation of InceptionV3 supports values from -1.0 - 1.0. 

For the targeted AdversarialPSO attack on the indices provided by the Parsimonious attack, execute:
```
python AdversarialPSO.py -d imagenet299 --maxChange 0.1 --blockSize 32 --pars -t -p 10
```

Note: maxChange was set to 0.1 because the Keras implementation of InceptionV3 supports values from -1.0 - 1.0. 


## Other options and usage parameters
A complete list of commandline arguments can be obtained by executing `python AdversarialPSO.py -h`
```
usage: AdversarialPSO.py [-h] [--dataset DATASET] [--maxChange MAXCHANGE]
                         [--numOfParticles NUMOFPARTICLES] [--targeted]
                         [--C1 C1] [--C2 C2] [--Samples SAMPLES] [--Randomize]
                         [--verbose VERBOSE] [--topN TOPN] [--sample SAMPLE]
                         [--blockSize BLOCKSIZE] [--Queries QUERIES] [--pars]

PSO Parameters

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Supports CIFAR10, MNIST, imagenet224, and imagenet299
  --maxChange MAXCHANGE
                        Controls the L-infinity distance between the source
                        and destination images
  --numOfParticles NUMOFPARTICLES, -p NUMOFPARTICLES
                        Number of particles in the swarm
  --targeted, -t        Choose random target when crafting examples
  --C1 C1               Controls exploitation weight
  --C2 C2               Controls explorations weight
  --Samples SAMPLES, -n SAMPLES
                        Number of test Samples to attack
  --Randomize           Randomize dataset
  --verbose VERBOSE, -v VERBOSE
                        Verbosity level. 0 for no terminal logging, 1 for
                        samples results only, and 2 for swarm level verbosity
  --topN TOPN           Specify the number of labels to reduce when attacking
                        imagenet
  --sample SAMPLE       Specify which sample to attack
  --blockSize BLOCKSIZE
                        Initial blocksize for seperating image into tiles
  --Queries QUERIES, -q QUERIES
                        Mazimum number of queries
  --pars                Run in Parsimonious... samples
  ```
  
For Imagenet, the topN argument specifies how many of the top N labels to attack. For example, `--topN 3` searches for inputs that reduce the model's confidence in the top 3 labels.


**YMMV: Due to some random aspects in the search process, results may vary.**
