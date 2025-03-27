## installation guide

```bash
# optional: if compressed folder for the project is not available
git clone https://github.com/Charles-CongFu/IRobman_Final_Project.git
cd IRobman_Final_Project
git checkout main

# clone pybullet-object-models within the project
git clone https://github.com/eleramp/pybullet-object-models.git

# install dependencies
# on macos
brew install uv
# on Ubuntu
sudo apt-get install uv
# on Arch Linux
sudo pacman -S uv

# install dependencies(listed in requirement.txt)
uv sync
uv pip install -r requirement.txt

# enable virtual environment
# bash
source .venv/bin/activate
# fish
source .venv/bin/activate.fish
```

## Usage

```bash
# run the main.py with different objects
python3 main.py --object YcbBanana
python3 main.py --object YcbFoamBrick
python3 main.py --object YcbHammer
python3 main.py --object YcbMediumClamp
python3 main.py --object YcbPear
python3 main.py --object YcbScissors
python3 main.py --object YcbStrawberry
python3 main.py --object YcbTennisBall
python3 main.py --object YcbGelatinBox
python3 main.py --object YcbMasterChefCan
python3 main.py --object YcbPottedMeatCan
python3 main.py --object YcbTomatoSoupCan
python3 main.py --object YcbCrackerBox
python3 main.py --object YcbMustardBottle
python3 main.py --object YcbChipsCan
python3 main.py --object YcbPowerDrill
```



## Codebase Structure

```shell

├── configs
│   └── test_config.yaml
├── main.py
├── README.md
└── src
    ├── objects.py
    ├── robot.py
    ├── simulation.py
    ├── utils.py
    │ # new features below
    ├── bounding_box
    │   └── bounding_box.py
    ├── grasping
    │   ├── grasp_execution.py
    │   ├── grasp_generation.py
    │   └── mesh.py
    ├── ik_solver
    │   └── ik_solver.py
    ├── obstacle_tracker
    │   └── obstacle_tracker.py
    ├── path_planning
    │   ├── planning_executor.py
    │   ├── potential_field.py
    │   ├── rrt_star_cartesian.py
    │   ├── rrt_star.py
    │   └── simple_planning.py
    └── point_cloud
        └── point_cloud.py
```