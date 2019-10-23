rem CALL "c:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat" -pyver 3.5

rem python .\shopper-gaze-monitor-python\main.py -m "c:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\Transportation\object_detection\face\pruned_mobilenet_reduced_ssd_shared_weights\dldt\INT8\face-detection-adas-0001.xml" -pm "c:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\Transportation\object_attributes\headpose\vanilla_cnn\dldt\INT8\head-pose-estimation-adas-0001.xml" -i cam -l "c:\Program Files (x86)\IntelSWTools\openvino\inference_engine\lib\intel64\Release\inference_engine.lib" -d GPU
python .\shopper-gaze-monitor-python\main.py -i cam -d GPU
