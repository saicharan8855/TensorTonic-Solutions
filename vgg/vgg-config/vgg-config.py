def make_vgg_config(variant: str) -> list:

    variant = variant.lower()

    if variant == "vgg11":
        return [64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"]
    if variant == "vgg16":
        return [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
    if variant == "vgg13":
        return [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    else:
        return [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"]
    """
    Returns the canonical VGG layer configuration as a new list.
    """
    pass