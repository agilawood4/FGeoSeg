import segmentation_models_pytorch as smp

def build_teacher_smp_unet_swin_tiny(num_classes=2, encoder_name='swin_tiny_patch4_window7_224', encoder_weights='imagenet'):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model
