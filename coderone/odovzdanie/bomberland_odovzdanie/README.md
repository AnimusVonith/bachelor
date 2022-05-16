# Jakub Adamčiak
# Bakalárska práca
# Posilňované učenie na hre Bomberman

Na spustenie je potrebné nainštalovať Docker.
https://docs.docker.com/get-docker/

Je potrebné si dať pozor a nainštalovať Docker compose v2, inak
sa nebude dať spustiť žiadny z definovaných kontajnerov.

Po nainštalovaní ho spustiť a v zložke, v ktorej sa nachádza tento
súbor, je potrebné otvoriť cmd (unix som neskúšal ale predpokladám,
že by to malo fungovať podobne) a zadať jeden z týchto príkazov:


# Zapnutie klienta v normálnom móde, spolu s buildom, 1. spustenie
docker-compose up --abort-on-container-exit --force-recreate --build --remove-orphans

# Zapnutie klienta v trénovacom móde, spolu s buildom, 1. spustenie (gym)
docker-compose -f open-ai-gym-wrapper-compose.yml up --force-recreate --abort-on-container-exit --build --remove-orphans


# Zapnutie klienta v normálnom móde
docker-compose up --abort-on-container-exit --force-recreate

# Zapnutie klienta v trénovacom móde
docker-compose -f open-ai-gym-wrapper-compose.yml up --force-recreate --abort-on-container-exit


Pre viac informácií, odporúčam pozrieť oficiálnu dokumentáciu tvorcov
tohto prostredia.
https://www.gocoder.one/bomberland

Na spustenie s inými parametrami treba zmeniť textový súbor `info.txt`, 
poprípade `settings.txt` v priečinku `python3`.