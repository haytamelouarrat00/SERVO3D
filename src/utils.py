import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_images(img1, img2, title1, title2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def get_rotation_matrix(roll, pitch, yaw):
    return R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
