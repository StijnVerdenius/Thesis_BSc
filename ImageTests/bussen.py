
from random import randint

def busMove(size, bus = []):

    kaarten = {}
    for x in range(13):
        kaarten[x] = 4


    drinks = 0
    buslength = size
    done = 0

    if (len(bus)!=size):

        for x in range(buslength):
            kaart = randint(0,12)
            while(kaarten[kaart] < 1):
                kaart = randint(0, 12)
            kaarten[kaart ] -= 1
            bus.append(kaart)

    while(done < buslength):
        huidigekaart = bus[done]
        gok = "H"
        if (huidigekaart > 6):
            gok = "L"
        kaart = randint(0, 12)
        while (kaarten[kaart] < 1):
            kaart = randint(0, 12)
        kaarten[kaart] -= 1


        if(kaart == huidigekaart):
            drinks += 2
            bus[done] = kaart
            done = 0
        elif((kaart > huidigekaart and gok == "H") or (kaart< huidigekaart and gok=="L")):
            bus[done] = kaart
            done += 1
        else:
            drinks += 1
            bus[done] = kaart
            done = 0

        controlecounter = 0
        for key in kaarten:
            controlecounter += kaarten[key]
        if controlecounter == 0:
            return drinks + busMove(size, bus = bus)

    return drinks

steekproef = 10000
print("steekproefgrootte: ", steekproef)
for k in range(10):
    teller =0
    max = 0
    min = 100000
    for x in range (steekproef):
        move = busMove(k)
        if (move > max):
            max  = move
        if (move < min):
            min = move
        teller += move
    print("busgrootte: ", k, " gemiddelde aantal drankjes: ", teller/steekproef, " max: ", max, " min: ", min)