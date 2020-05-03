# Traffic Sign Adversarial Attack

This is a project discussing the ethical issues regrading the possibility of adversarial attack on  self-driving car system.

The report is open to review for Harvard community [Adversarial Attacks on Traffic Sign Recognition System for Autonomous Vehicles](https://docs.google.com/document/d/1x-QusFRZBhfNtKUmDs824f-UmMYUe34NBhCS6YIdapI/edit?usp=sharing).

## Dataset

The GTSRB dataset (German Traffic Sign Recognition Benchmark) is provided by the Institut für Neuroinformatik group [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). It was published for a competition held in 2011 ([results](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results)). Images are spread across 43 different types of traffic signs and contain a total of 39,209 train examples and 12,630 test ones.

## generate adversarial examples

Run the script fooler_generator.py to generate adversarial examples

  ```sh
  python fooler_generator.py
  ```
  
  You can include one or more command line arguments.

  ```sh
  python fooler_generator.py --help
  ```
  
 The training process is shown in model_training.ipnyb. The attacking evaluation is displayed in the advattack.ipnyb.
  
  
