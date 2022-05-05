from pytorch_pretrained_vit import ViT
def vit(pretrained=True):
    return ViT('B_16_imagenet1k', pretrained=True, image_size=640)
