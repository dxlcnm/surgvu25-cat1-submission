## 2rd Place Solution - Surgvu2025 Category1: Surgical tool classification and localization (MICCAI 2025)

A Practical Two-Stage Method for Surgical Tool Classification and Localization
## Challenge
You can find the challenge page and the final result [here](https://surgvu25.grand-challenge.org/surgvu/)

## Models download
Download the models to the `model` directory. See the details about [Model Download](model/README.md)

## Installation
`pip install -r requirement.txt`

## Inference
The inference example:
<br>`python inference.py --input_path test/input/interf0 --output_path test/output/interf0`

The output result is saved in a json format file like this
```json
{
  "name": "Regions of interest",
  "type": "Multiple 2D bounding boxes",
  "boxes": [
    {
      "name": "slice_nr_21_force_bipolar",
      "corners": [
        [0, 2, 0.5],
        [146, 2, 0.5],
        [146, 113, 0.5],
        [0, 113, 0.5]
      ],
      "probability": 0.26842018961906433
    }
  ]
}
```
## Docker
```
bash

# build
bash do_build.sh

# test
bash do_test_run.sh

# save
bash do_save.sh
```
The docker run example after docker build:<br>
`docker run --gpus all -v $(pwd)/test/input/interf0:/input -v $(pwd)/test/output/interf0:/output -it medibot-category-1-final-phase:latest`


## Code
The training code is derived from the following repo:<br>
[1] [https://github.com/ultralytics/ultralytics.git](https://github.com/ultralytics/ultralytics.git)<br>
[2] [https://github.com/Tianfang-Zhang/CAS-ViT.git](https://github.com/Tianfang-Zhang/CAS-ViT.git)