"""
Image augmentation configuration wrapper.
"""

class ImageAugmentationConfig:
    """
    Encapsulate an augmentation configuration.
    Example:

        config = ImageAugmentationConfig()
        config.rotation([20, 80])
        config.flip()
    """
    def __init__(self):
        self.do_rotation = False
        self.do_flip = False

    def rotation(self, angles):
        """
        Add rotations.
        """
        self.do_rotation = True
        self.rotation_angles = angles

    def flip(self):
        """
        Add flip transformation.
        """
        self.do_flip = True

