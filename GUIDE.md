## Prerequisites

python 3.8

## Installation Steps

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Clone the pybullet-object-models repository:
```bash
git clone https://github.com/eleramp/pybullet-object-models.git
```

3. Install the pybullet-object-models package:
```bash
pip install -e pybullet-object-models/
```

## Codebase Structure

```shell

├── configs
│   └── test_config.yaml # config file for your experiments (you can make your own)
├── main.py # example runner file (you can add a bash script here as well)
├── README.md
└── src
    ├── objects.py # contains all objects and obstacle definitions
    ├── robot.py # robot class
    ├── simulation.py # simulation class
    ├── utils.py # helpful utils
    │ # new features below
    ├── grasping
    │   ├── grasping_mesh.py
    │   └── grasping.py
    ├── path_planning
    │   ├── planning_executor.py
    │   ├── rrt_star_cartesian.py
    │   ├── rrt_star.py
    │   └── simple_planning.py
    ├── point_cloud
    │   ├── bounding_box.py
    │   ├── object_mesh.py
    │   └── point_cloud.py
    ├── ik_solver.py
    └── obstacle_tracker.py
```