# Special Topics in Programming for Performance and Installation
This project implements a real-time visual feedback loop that explores the concepts of system hysteresis and predictive rendering. It uses a decoupled architecture: TouchDesigner handles the real-time video ingestion and rendering, while an external Python process calculates the Optical Flow (using OpenCV's Farneback algorithm) to "predict" and warp the next frame. These two communicate with zero-latency using the NDI protocol.

## Prerequisites 
* **OS:** macOS or Windows.
* **TouchDesigner:** Installed
* **NDI Tools:** Download and install the core NDI drivers for your OS from the official NDI website. https://ndi.video/tools/
* **Python Environment Manager:** Miniforge or Anaconda.

## Approach of using

### Step 1: Python Environment Setup 
Recommended to run the prediction engine in a dedicated virtual environment.

Open your Terminal and run the following commands:

1. Create and activate the environment:
```
conda create -n td_ai_bridge python=3.10 -y 
conda activate td_ai_bridge
```

Install required libraries:

```
pip install numpy opencv-python 
pip install ndi-python
```
### Step 2: The Python Prediction Engine (bridge.py)
Using the file named bridge.py in your project folder.

### Step 3: TouchDesigner Network Setup
Send Video Out:

Open the TD(.toe) file,Open your Terminal, ensure your environment is activated (conda activate td_ai_bridge), and navigate to your project folder.

Run the script: ```python bridge.py ```

Wait until the terminal outputs: "Running!"
Receive Predicted Video:

Back in TouchDesigner, click the "NDI In TOP".In the parameter window, click the dropdown for Source Name and select the Python stream (it will look like YourComputerName (Python_Future)).

Then it works!

## Inspiration
The inspiration for this project stems from reflections on algorithmic bias—specifically, machine learning models derived by the results based entirely on the underlying composition of their training datasets and databases. Combined with my observations on how political imagery(https://thenewinquiry.com/sci-fi-crime-drama-with-a-strong-black-lead/) is constructed and manipulated, this sparked a deep interest in the discrepancies between action prediction models (commonly used in computer vision) and actual real-world dynamics. While the profound implications of algorithmic bias are vast and warrant deep exploration, my current focus is on materializing this concept through a technical workflow base on latency between reality and prediction.

## Technical
My initial concept was to utilize PredNet (https://github.com/coxlab/prednet) —a deep predictive coding network—to process live video frames, generate "predicted future" images, and route this data back into TouchDesigner. In this setup, the actual live webcam feed would be treated as the "delayed reality," allowing me to conduct visual experiments based on the friction and divergence between the AI's hallucinated future and the lagging physical present.

However, during research and testing, I discovered that running complex neural networks locally in real-time suffers from severe frame rate instability and computational latency. Therefore, as a pragmatic prototype, I pivoted to using Optical Flow (a physics-based prediction model) for this iteration. Although the conceptual implications shift slightly—from a dataset-driven cognitive bias to a physics-driven inertial bias—the deployment architecture, the pipeline, and the resulting visual artifacts remain consistent to a large extent.

This project utilizes Dense Optical Flow (specifically the Farneback algorithm via OpenCV) as its core prediction engine. By analyzing the brightness constancy between two consecutive frames, the algorithm calculates a dense 2D vector field that maps the instantaneous velocity and direction (u,v) of every single pixel.
optical flow： https://en.wikipedia.org/wiki/Optical_flow

currently using the default parameters, can be fine-tuned

In Python
```
flow = cv2.calcOpticalFlowFarneback( prev_gray, curr_gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0 )
```
## Further reflection

Future iterations could focus on an in-depth study of the technical mechanics behind predictive models, seeking to push them into deeper technical implementations while justifying their applied rationale in Creative Expression.


