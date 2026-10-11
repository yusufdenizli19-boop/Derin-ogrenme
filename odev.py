w1 = 0.5
w2 = 0.3

batch_size = 1
epoch_size = 1

hedef = 0

while True:
    sonuc = w1 + w2 - 0.8
    print(sonuc)

    if sonuc == hedef:
        break

    w1 = w1 - sonuc / 2
    w2 = w2 - sonuc / 2