

def roundRatings (value, resolution=0.5):
    rating =  round (value / resolution) * resolution
    if rating < 0.5:
        return 0.5
    elif rating > 5.0:
        return 5.0
    else:
        return rating
