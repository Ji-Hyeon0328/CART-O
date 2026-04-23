# CARTO Low-Level Controller MATLAB Demo

This folder contains MATLAB prototypes for the low-level controller design of the CARTO framework.

## Overview
The current pipeline is organized as:

High-level meta command  
→ Theta Decoder  
→ Theta-to-Reference Mapper  
→ Force MPC  
→ WBC  
→ Impedance Residual  
→ Final Torque Output

The goal of these demos is to verify that terrain-aware high-level signals can be translated into executable low-level gait, reference, and control parameters.

## Main Components
- **Theta Decoder**  
  Decodes high-level signals such as gait mode, meta gait action, and command/state information into structured low-level parameters:
  - gait parameters
  - foot parameters
  - base references
  - control parameters

- **Theta-to-Reference Mapper**  
  Converts decoded parameters into:
  - contact schedule
  - base reference trajectory
  - foot reference trajectory

- **Force MPC**  
  Generates desired ground reaction force (GRF) trajectories over the horizon.

- **WBC**  
  Produces nominal torque commands from the force/reference information.

- **Impedance Residual**  
  Adds contact-aware residual correction, especially to reduce touchdown impact and improve robustness.

## Current Status
The OC block has been implemented and tested in simple MATLAB demos.

Verified pipeline:
- High-level meta plan decoding
- Reference generation
- MPC → WBC torque generation
- Impedance residual attachment
- Final torque output

Simple demos have been tested for:
- Go1
- Spot

## Included Demo Types
This folder includes prototypes for:
- single-robot low-level controller tests
- Go1 / Spot preset-based tests
- virtual terrain tests
- mixed-terrain adaptation demos

## Important Note
These MATLAB scripts are currently intended for **concept validation and controller structure testing**.

They do **not yet** represent full closed-loop simulation or real-robot validation.

In particular:
- WBC is currently a robot-scaled mock version
- robust / CBF blocks are not included yet
- true energy-efficiency claims require closed-loop simulation or real-robot experiments with mechanical work / CoT evaluation

## Planned Next Steps
- refine WBC with more realistic robot dynamics
- add robust / CBF blocks
- evaluate energy-aware behavior
- validate in simulation and eventually on real hardware

## Author
Ji-Hyeon Yoo
