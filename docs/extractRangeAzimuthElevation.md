# Extract Range-Elevation-Azimuth Heatmap from Raw mmWave Signals


## Prerequisite

AWR1642EVM-ODS + DCA1000 data capture setup. Capture raw mmWave data (adc_data.bin) and save in `./data/data_raw/[posture]/`. You may debug first with my [data](https://drive.google.com/file/d/16AnJaEJpsRUfh3Qct37vUvPs4iKCM5G9/view?usp=sharing).

## Run & Explanation

- Open `./mmWave_py/extractHeatmaps.py` and spicify path to data in `./data/`. 
- Run the first block in `./mmWave_py/extractHeatmaps.py` to extract heatmaps from mmWave frames.

    1. The core function is `featureExtraction` in `utility.py`. This function takes two inputs (data folder, output folder) and outputs heatmapa. 
    
    2. Data is being processed with the following steps.
       
        - Load `.bin` files. Set hardcoded Radar config parameters (see below). The configure is from your settings in mmWave Studio. My config is in `./data/config/mPose0714.xml`.
        ```
        # line 250
        numFrames = 640
        numADCSamples = 512
        numTxAntennas = 2
        numRxAntennas = 4
        numLoopsPerFrame = 76
        numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
        
        BINS_PROCESSED = 55
        ANGLE_RES = 1
        ANGLE_RANGE = 70
        ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
        
        dig_out_sample_rate=6000 
        freq_slope_const=39.010
        ```
        - Get steering vector for further beamforming.
        - Traverse each range bin, separate data to match with virtual antenna setup.
        - For each range, conduct beamforming. (We get 2 Range-Azimuth and 1 Range-Elevation heatmaps in this step.)
        - Prescale heatmap intensity as they are too high. (I do this for further saving them as Float32 without losing trends.)
        - Union 3 heatmaps as one (squarish). Save heatmaps and timestamps. 
        - Write to output path, return.

- Run the second block to visualize a sample of the extracted 3 in 1 heatmaps.
