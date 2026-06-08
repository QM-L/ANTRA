# ANTRA: Ablation Needle Trajectory Advisor

Antra is a system meant for liver ablation procedures during the needle planning phase. It will segment, analyze and extrapolate from a CT-scan dicom, outputting possible needle trajectory from a user defined ablation center.

### Usage

At the moment, Antra is not yet fit for compilation. Running the python script raw or via the bat files is the only tested form of usage. This will run the tumor selection demo followed by a visualisation of raytracing. This might take ~1 minute. On the first run, segmentation will be ran, which might take ~10 minutes without a CUDA-enabled GPU.
1. Clone the repository
```
git clone https://github.com/QueasyQuery/OCKY-bot
```
2. Run `setup.bat` once on download (installs dependencies)
3. Put a dicom folder of a ct-scan into `/scans/`. 
4. Run `run_tool.bat` and enter the name of the dicom folder.

To test repeatedly without having to enter a dicom folder each time, change the variable `DEFAULT_DICOM` in `main.py` and run `run_raw.bat`. 

## Acknowledgements

 - [TotalSegmentator](https://github.com/wasserth/totalsegmentator)