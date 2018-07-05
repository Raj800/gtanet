def truncate(steering):
    if steering < 0:
        steering = 0
    if steering >= 1000:
        steering = 999
    if 0 <= steering < 100:
        steering = 0
    elif 100 <= steering < 200:
        steering = 1
    elif 200 <= steering < 300:
        steering = 2
    elif 300 <= steering < 350:
        steering = 3
    elif 350 <= steering < 400:
        steering = 4
    elif 400 <= steering < 420:
        steering = 5
    elif 420 <= steering < 440:
        steering = 6
    elif 440 <= steering < 460:
        steering = 7
    elif 460 <= steering < 470:
        steering = 8
    elif 470 <= steering < 480:
        steering = 9
    elif 480 <= steering < 490:
        steering = 10
    elif 490 <= steering < 495:
        steering = 11
    elif 495 <= steering < 496:
        steering = 12
    elif 496 <= steering < 497:
        steering = 13
    elif 497 <= steering < 498:
        steering = 14
    elif 498 <= steering < 499:
        steering = 15
    elif 499 <= steering < 500:
        steering = 16
    elif 500 <= steering < 501:
        steering = 17
    elif 501 <= steering < 502:
        steering = 18
    elif 502 <= steering < 503:
        steering = 19
    elif 503 <= steering < 504:
        steering = 20
    elif 504 <= steering < 505:
        steering = 21
    elif 505 <= steering < 510:
        steering = 22
    elif 510 <= steering < 520:
        steering = 23
    elif 520 <= steering < 530:
        steering = 24
    elif 530 <= steering < 540:
        steering = 25
    elif 540 <= steering < 560:
        steering = 26
    elif 560 <= steering < 580:
        steering = 27
    elif 580 <= steering < 600:
        steering = 28
    elif 600 <= steering < 650:
        steering = 29
    elif 650 <= steering < 700:
        steering = 30
    elif 700 <= steering < 800:
        steering = 31
    elif 800 <= steering < 900:
        steering = 32
    elif 900 <= steering < 1000:
        steering = 33
    else:
        steering = 34
    return steering

def detruncate(steering):
    if steering == 0:
        steering = 50
    if steering == 1:
        steering = 150
    if steering == 2:
        steering = 250
    if steering == 3:
        steering = 325
    if steering == 4:
        steering = 375
    if steering == 5:
        steering = 410
    if steering == 6:
        steering = 430
    if steering == 7:
        steering = 450
    if steering == 8:
        steering = 465
    if steering == 9:
        steering = 475
    if steering == 10:
        steering = 485
    if steering == 11:
        steering = 493
    if steering == 12:
        steering = 495
    if steering == 13:
        steering = 496
    if steering == 14:
        steering = 497
    if steering == 15:
        steering = 498
    if steering == 16:
        steering = 499
    if steering == 17:
        steering = 500
    if steering == 18:
        steering = 501
    if steering == 19:
        steering = 502
    if steering == 20:
        steering = 503
    if steering == 21:
        steering = 504
    if steering == 22:
        steering = 505
    if steering == 23:
        steering = 507
    if steering == 24:
        steering = 515
    if steering == 25:
        steering = 525
    if steering == 26:
        steering = 535
    if steering == 27:
        steering = 550
    if steering == 28:
        steering = 570
    if steering == 29:
        steering = 590
    if steering == 30:
        steering = 625
    if steering == 31:
        steering = 675
    if steering == 32:
        steering = 750
    if steering == 33:
        steering = 850
    if steering == 34:
        steering = 950
    return steering
