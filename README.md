# master-thesis-python
This repository contains the processing of the raw data obtained in the master thesis called "Exploring the Influence of Visual, Auditory, and Mental Stimuli on Cognitive Load in a Virtual Shopping Environment". This readme explains the structure of the code and can be used as a guide for understanding how to use it. It consists of two parts: the preprocessing part and the data analysis part. In the preprocessing part, the data is filtered so that it could be used to plot the data. In the data analysis part, the data is plotted and statistically analyzed. 

## Setting up the project
First, install the requirements in your virtual environment:
```pip install requirements.txt```

Second, make sure everything is set correctly. The code only works with the raw data, which can be found in the TU Delft project directory. Moreover, some parts require very long computation times. In order to overcome this issue, I added pickle files in the TU Delft project directory containing the processed data for the functions. Set the data file correctly:
1. Create a data directory in the project at the same level as your \src directory.
2. Add all raw data from the participants in the data directory. You should have 22 participant directories in the data directory.
3. Add the pickles in the data directory. Now you have 23 directories, 22 for the participants and 1 pickle directory.
4. Add the _nasa-tlx-results.xlsx_ and _participants-conditions.xlsx_ file in a directory in the data directory called _nasa_tlx_ 

Now add the environment variables so that the functions can find your data directory. 
1. In your project directory, add a ".env" file.
2. Add the following variables, according to your wishes:
   ```
      DATA_DIRECTORY = C:\path\to\your\data\directory
      ECG_SAMPLE_RATE = 1024
      EDA_SAMPLE_RATE = 32
      NASA_TLX_FILENAME = nasa-tlx-results.xlsx
      PERFORMANCE_FILENAME = performance-results.xlsx
   ```

Your project is ready to go!

## Preprocessing
The preprocessing code could be found in the preprocessing directory. Each measure is calculated in a different directory, since they often use different types of data. The data is divided into different directories:
1. The data from the head-mounted display (hmd), is preprocessed in the hmd directory, and consists of the performance, area of interest, and movements. 
2. Data from the ECG/EDA device is processed in the ecg_eda directory, which is divided into ECG and EDA. EDA has not been used for the researched, and is therefore not completely finished.
3. The NASA-TLX questionnaire is processed in the nasa_tlx directory. 
4. The preprocessing of all these measures is combined together in one large dataframe, created in the main_dataframe directory. This dataframe is always used in the data analysis part, and could be obtained using the **main_dataframe.pickle** file. Also, the **main_dataframe_long.pickle** has been used, which contains the same dataframe as main_dataframe.pickle, but in a long format. 
5. Lastly, some helper functions are created that do not completely belong to one of the features. These functions are more general functions that could be used in all these preprocessing directories. 

## Data analysis
The data analysis is divided into different directories:
1. The most used directory is the visualization directory, which contains all the visualization scripts for the plots. The name of the script file describes which plots could be created in that script. Most pickle files are used to plot right away; without the pickle files, the plots may take a very long time to compile. To run the plots, run the function at the bottom of each script in the __main__ section.
2. The statistical analysis part uses the main_dataframe to compute statistical values such as correlation, cohen's d, and other statistical checks.
3. Also this directory contains some helper function, which support the visualization and statistical analysis scripts.

Happy coding! If you have any questions, feel free to contact me. 

Best, 
Job




