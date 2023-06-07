# Multi-Agent Dynamic Police Patrol Dispatching and Rescheduling Problem (MADPRP)

This is an implementation code for our paper entitled Learning to Send Reinforcements: Coordinating Multi-Agent Dynamic Police Patrol Dispatching and Rescheduling via Reinforcement Learning to be published in IJCAI 2023.

## Supplementary Material
The supplementary material for the above-mentioned paper can be found in ```Learning_To_Send_Reinforcements_Supplementary_Material.pdf```. It contains detailed descriptions of the data, experiment settings and the implemented code.

## Step-By-Step Guide

### Preprocessing Steps

1. Refer to  ```./preprocess/PreprocessGeoFile.py```
    - This file takes is in ```./data/RawData/road_name_grid0.02_npc2.geojson```.
    - It outputs ```grids_medoid.pkl```, ```grids_nodes_count.pkl``` and ```travel_time_matrix.pkl```.
    - All the output files are saved in ```./data/RawData```.

2. Refer to ```./preprocess/PreprocessRawData.py```
    - A function ```process_patrol_areas_files()``` takes in all the raw data files to output a Dataset Object.
    - The filenames of all the raw data files can be found in ```./constants/Settings.py```.
    - The structure of the Dataset object is found in ```./entity/Dataset.py```.
    - Dataset object consists of Sector objects representing each patrol sector. The structure of the Sector object can be found in ```./entity/Sector.py```.
    - Each sector consists of multiple patrol agents and patrol areas.

3. Refer to ```PreProcessInputData.py```
    - This file calls a function ```process_patrol_areas_files()``` found in ```./preprocess/PreprocessRawData.py```.
    - It outputs a Dataset object and saved as ```./data/processed_data.pkl```.

4. Refer to ```InitialScheduleSampleGenerator.py```
    - This file outputs sample initial schedules (by individual sector) to be used for training or testing.
    - This file solves a SetCover model (found in ```./model/SetCover.py```) to optimality using CPLEX.
    - All training and testing files are stored in ```./data/Training``` and ```./data/Testing``` respectively.


Note: 
- ```./data/processed_data.pkl``` is the main input file containing the setup of the patrol sectors environment.
- ```./data/Training``` and ```./data/Testing``` contain the samples of initial schedules.
- ```2``` or ```4``` in folder names indicate the poisson mean used to generate the incident.

### Parameter Settings

```./constant/Settings.py``` contain parameters for various aspects of the model. You can either fix the parameter in this file or allow user to modify 
the parameter via passing an argument through command line.

### Basic Building Blocks

- ```./entity``` folder contains the basic classes used in the model. 
- Dataset > Sector > PatrolArea represents the hierarchy of the patrol environment.
- Incident class to represent an incident instance.
- Defect class to represent an defect instance (see the earlier paper on the definition of defects in the context of schedule).
- Schedule class to represent a schedule of an individual sector. Each schedule contain schedules of each sectoral agents/patrol agents.

Note: New classes to represent a basic building block should be declared here.

### Utility Functions

1. Refer to ```/utils/ScheduleUtil.py```
    - This file contains the utitiliy functions pertaining to schedule.

2. Refer to ```./utils/util.py```
    - This file contains the common utility functions used throughout the model.

3. Refer to ```./data/ScenarioGenerator.py```
    - This file generates a scenario i.e. dynamic occurences of incident as incident object.
    - The parameters needed to produce the scenario statistically can be found in ```./constant/Settings.py```.
    - The function ```compute_location_pdf()``` computes the probability distribution of the occurences of incident by each patrol area in a given sector. It is run once and the outputs are saved as ```./data/location_pdf_<sector>.pkl```.

Note: You may want to put new utility functions here.

### Non-RL Models

All non-RL models can be found in ```./model```.

1. Refer to ```ReschedulerMA.py``` 
    - Contains the rescheduling heuristics with ejection chain for multi-agent DPRP

2. Refer ```Simulator.py``` 
    - This file contains the basic simulator model to simulate one-day shift given an initial schedule and dynamic occurences of incidents.
    - The learning agent/learning model/policy is inputted to the simulator object to dictate the action needed whenever dynamic event occurs.

### RL Models

- Separate folders created for each of the RL models such as ```vfa``` and ```mavfa```.
- Each folder contains parameter folder to store the learnt parameters, a network file to build the network architecture, agent file for the learing algorithm and 
training file for the training procedure can be found in ```./data/Training/```.

### Training

1. Refer to ```Trainer.py```
    - Take in input data file and calls the training function of selected RL models.

2. Refer ```./xxx/XXXTrain.py```
    - Contains the training procedure for xxx model.
    - The learnt parameters are saved in ```./xxx/parameters/``` while the output files are saved in ```./xxx/output/```.
    - To allow replication of training or using the same training dataset to train a different model, include ```--save=True``` in the command line to save the training instances 
    and ```--replicate=True``` to use the same training instances.
    - The training instances are saved in ```./data/Training/instances/<poisson mean>/xxx/training_instances_<sectors>.pkl```.
    - Each training instances contain the training scenarios and the corresponding results when running a myopic approach (i.e. running rescheduling heuristic with ejection chain 
    w/o any consideration for future value).
    - The output of the training include ```loss_by_step.pkl, presence_<sectors>.pkl```, ```success_<sectors>.pkl``` and ```respond_<sectors>.pkl``` containing the loss values, the presence utility value, success rate and response rate of each training episode measured as % improvement over myopic.

### Running Experiments / Testing

Refer to ```RunExperiment.py```
- Take in the learned model, run the experiments by calling the simulator instances.
- Each experiment contains a given initial schedule (from ```./data/Testing``` folder) while each scenarios of a given experiment contain various possibility of occurences of dynamic events.
- Output the results and computation time by each incident in a given experiment and scenario. Results is a tuple of ```<response, success and presence>```
(represented as absolute values and not as an improvement over myopic).
- To allow replication of testing or using the same testing dataset to test a different model, include ```--save=True``` in the command line to save the testing instances 
and ```--replicate=True``` to use the same testing instances.
- The testing instances are saved in ```./experiment/test_cases/test_cases_<sectors>_<experiment_idx>.pkl```.

### Sample Run command

```
python Trainer.py --sectors=EFL --model=VFA--poisson_mean=2 --pre_trained=True
python RunExperiment.py --sectors=EFL --model=greedy
```
Note: 
- If only one sector is inputted, the model should solve it as if it is a single-agent problem.
- You may input the following models in the ```--model=```: greedy, myopic, VFA or MADQN.
- Ensure thate ```pre_trained=True``` if you are running learned models. Make sure that the parameter file is located at the corresponding parameter subfolder (for e.g. ```./xxx/parameters/```).

## References
