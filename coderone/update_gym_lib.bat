@ECHO OFF
del bomberland_2\python3\gym_lib.py
copy bomberland\python3\gym_lib.py bomberland_2\python3\gym_lib.py
del bomberland_3\python3\gym_lib.py
copy bomberland\python3\gym_lib.py bomberland_3\python3\gym_lib.py
del bomberland_4\python3\gym_lib.py
copy bomberland\python3\gym_lib.py bomberland_4\python3\gym_lib.py
echo update done