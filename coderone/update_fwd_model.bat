@ECHO OFF
del bomberland_2\python3\forward_model.py
copy bomberland\python3\forward_model.py bomberland_2\python3\forward_model.py
del bomberland_3\python3\forward_model.py
copy bomberland\python3\forward_model.py bomberland_3\python3\forward_model.py
del bomberland_4\python3\forward_model.py
copy bomberland\python3\forward_model.py bomberland_4\python3\forward_model.py
echo update done