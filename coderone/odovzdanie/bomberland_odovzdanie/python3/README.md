# Jakub Adamčiak
# Bakalárska práca
# Posilňované učenie na hre Bomberman

`agent.py` 				-	pripojenie ku klasické hernému serveru
`dev_gym.py` 			-	otvorí ai gym wrapper - trénovací mód (agent)
`gym_lib.py`			-	používané gym wrapperom - trénovací mód (env)
`info.txt`				-	nastavenia pre model 
						-	oddelene novym riadkom: vyber nn, vyber algoritmu (ppo or a2c), meno modelu
`settings.txt`			-	nastavenia pre proces trénovania
						-	podporuje len 4 nastavenia, ktore su v subore definovane
						-	pre pokrocilejsie upravy treba ist do zdrojovych textov
`misc_funcs.py`			-	funkcie používané agentom
`model_funcs.py`		-	načítanie alebo vytvorenie nového RL modelu
`nn_setup.py`			-	definície neurónových sietí
`shaping_funcs.py`		-	tvarovanie pozorovania
`arena_templates.py`	-	príklady arén