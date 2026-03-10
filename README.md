# Special Topics in Programming for Performance and Installation

This project implements a real-time visual feedback loop that explores the concepts of system hysteresis and predictive rendering. It uses a decoupled architecture: TouchDesigner handles the real-time video ingestion and rendering, while an external Python process calculates the Optical Flow (using OpenCV's Farneback algorithm) to "predict" and warp the next frame. These two communicate with zero-latency using the NDI protocol.

@Prerequisites OS: macOS or Windows.

TouchDesigner: Installed

NDI Tools: Download and install the core NDI drivers for your OS from the official NDI website. https://ndi.video/tools/

Python Environment Manager: Miniforge or Anaconda.

@Approach of using

Step 1: Python Environment Setup recommended to run the prediction engine in a dedicated virtual environment.

Open your Terminal and run the following commands:

1.Create and activate the environment:

Bash 
conda create -n td_ai_bridge python=3.10 -y 
conda activate td_ai_bridge

2.Install required libraries:

Bash 
pip install numpy opencv-python 
pip install ndi-python

@ Step 2: The Python Prediction Engine (bridge.py) using the file named bridge.py in your project folder .

@ Step 3: TouchDesigner Network Setup Send Video Out:

open the TD(.toe) file
Open your Terminal, ensure your environment is activated (conda activate td_ai_bridge), and navigate to your project folder.
Run the script: python bridge.py.
Wait until the terminal outputs: "Optical Flow Engine Running!"
Receive Predicted Video:
Back in TouchDesigner, click the NDI In TOP.
In the parameter window, click the dropdown for Source Name and select the Python stream (it will look like YourComputerName (Python_Future)).
then it will show !



@Inspiration

The inspiration for this project stems from reflections on algorithmic bias—specifically, how machine learning models derive their results based entirely on the underlying composition of their training datasets and databases. Combined with my observations on how political imagery is constructed and manipulated, this sparked a deep interest in the discrepancies between action prediction models (commonly used in computer vision) and actual real-world dynamics. While the profound implications of algorithmic bias are vast and warrant deep exploration, my current focus is on materializing this concept through a technical workflow.

@Technical

My initial concept was to utilize PredNet（https://github.com/coxlab/prednet?tab=readme-ov-file）—a deep predictive coding network—to process live video frames, generate "predicted future" images, and route this data back into TouchDesigner. In this setup, the actual live webcam feed would be treated as the "delayed reality," allowing me to conduct visual experiments based on the friction and divergence between the AI's hallucinated future and the lagging physical present.

However, during research and testing, I discovered that running complex neural networks locally in real-time suffers from severe frame rate instability and computational latency. Therefore, as a pragmatic prototype, I pivoted to using Optical Flow (a physics-based prediction model) for this iteration. Although the conceptual implications shift slightly—from a dataset-driven cognitive bias to a physics-driven inertial bias—the deployment architecture, the pipeline, and the resulting visual artifacts remain consistent with my original vision.

This project utilizes Dense Optical Flow (specifically the Farneback algorithm via OpenCV) as its core prediction engine. By analyzing the brightness constancy between two consecutive frames, the algorithm calculates a dense 2D vector field that maps the instantaneous velocity and direction (u,v) of every single pixel.

currently using the default parameters, can be fine-tuned.
flow = cv2.calcOpticalFlowFarneback( prev_gray, curr_gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0 )

optical flow： https://en.wikipedia.org/wiki/Optical_flow
