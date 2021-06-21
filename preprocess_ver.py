def get_win_slideby(gcoh_ver):
    if gcoh_ver == 1:
        win = 256
        slideby = 128
    elif gcoh_ver == 2:
        win = 512
        slideby = 128
    elif gcoh_ver == 3:
        win = 256
        slideby = 64
    elif gcoh_ver == 4:
        win = 384
        slideby = 128
    elif gcoh_ver == 5:
        win = 768
        slideby = 128
    elif gcoh_ver == 6:
        win = 512
        slideby = 64
    elif gcoh_ver == 7:
        win = 2048
        slideby = 512
    elif gcoh_ver == 8:
        win = 1024
        slideby = 256
    elif gcoh_ver == 9:
        win = 384
        slideby = 64
    elif gcoh_ver == 10:
        win = 192
        slideby = 64
    elif gcoh_ver == 11:
        win = 192
        slideby = 192
    elif gcoh_ver == 12:
        win = 256
        slideby = 256
    elif gcoh_ver == 13:
        win = 384
        slideby = 384
    elif gcoh_ver == 14:
        win = 256
        slideby = 32
    elif gcoh_ver == 15:
        win = 1024
        slideby = 256

    return win, slideby
