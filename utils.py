import numpy as np
from matplotlib.patches import Ellipse

def generate_bounding_box_from_mask(mask):
    """
    Function for generating a bounding box around a mask.
    Bounding box covers the extremes of the mask inclusively such that the far left box aligns
    with the far left of the mask.
    :param mask: 2D mask image (zero and non-zero pixels). Non-zero pixels counted as wanted (True) pixels
    :return: List of inclusive bounding box coordinates. Format [<left>, <top>, <bottom>, <right>]
    """
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError("No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]

def draw_cov(box, covs, ax, colour='k', mode='ellipse'):
    """
    Function for drawing covariances around a bounding box
    :param box: list of box corner coordinates. Format: [<left>, <up>, <right>, <bottom>]
    :param covs: covariance matrices for both corners. Format: [<top_right_covariance>, <bottom_right_covariance>].
    Individual covariances format: [[<x_variance>, <xy_correlation>], [<xy_correlation>, <y_variance>]]
    :param ax: Matplotlib axis where covariances shall be drawn.
    :param colour: Colour for the covariance to be drawn in. Default: Black
    :param mode: Mode of covariance drawing. Options: ['arrow', 'ellipse']. Default: 'ellipse'
    :return: None
    """

    # Calculate the eigenvalues and eigenvectors for both top-left (tl) and bottom-right (br) corner covariances
    tl_vals, tl_vecs = np.linalg.eig(covs[0])
    br_vals, br_vecs = np.linalg.eig(covs[1])

    # flip the vectors along the y axis (for plotting as y is inverted)
    tl_vecs[:, 1] *= -1
    br_vecs[:, 1] *= -1

    # Determine which eigenvalue/vector is largest and which covariance that corresponds to
    tl_val_max_idx = np.argmax(tl_vals)
    br_val_max_idx = np.argmax(br_vals)
    argmax_cov1 = np.argmax(np.diag(covs[0]))
    argmax_cov2 = np.argmax(np.diag(covs[1]))

    # Calculate the magnitudes along each eigenvector used for visualization (2 * std dev)
    magnitude_tl1 = np.sqrt(covs[0][argmax_cov1][argmax_cov1]) * 2
    magnitude_br1 = np.sqrt(covs[1][argmax_cov2][argmax_cov2]) * 2
    magnitude_tl2 = np.sqrt(covs[0][np.abs(1 - argmax_cov1)][np.abs(1 - argmax_cov1)])*2
    magnitude_br2 = np.sqrt(covs[1][np.abs(1 - argmax_cov2)][np.abs(1 - argmax_cov2)])*2

    # Calculate the end-points of the eigenvector with given magnitude
    tl_end1 = tl_vecs[tl_val_max_idx] * magnitude_tl1 + np.array([box[0], box[1]])
    br_end1 = br_vecs[br_val_max_idx] * magnitude_br1 + np.array([box[2], box[3]])
    tl_end2 = tl_vecs[np.abs(1 - tl_val_max_idx)] * magnitude_tl2 + np.array([box[0], box[1]])
    br_end2 = br_vecs[np.abs(1 - br_val_max_idx)] * magnitude_br2 + np.array([box[2], box[3]])

    # Calculate the difference in the x and y direction of the corners
    tl_dx1 = tl_end1[0] - box[0]
    tl_dy1 = tl_end1[1] - box[1]
    tl_dx2 = tl_end2[0] - box[0]
    tl_dy2 = tl_end2[1] - box[1]

    br_dx1 = br_end1[0] - box[2]
    br_dy1 = br_end1[1] - box[3]
    br_dx2 = br_end2[0] - box[2]
    br_dy2 = br_end2[1] - box[3]

    # Draw the appropriate corner representation of Gaussian Corners
    if mode == 'arrow':
        # Draw the arrows only if they have magnitude
        if tl_dx1 + tl_dy1 != 0:
            ax.arrow(box[0], box[1], tl_dx1, tl_dy1, width=2, color=colour,
                     length_includes_head=True, head_width=6, head_length=2)
        if tl_dx2 + tl_dy2 != 0:
            ax.arrow(box[0], box[1], tl_dx2, tl_dy2, width=2, color=colour,
                     length_includes_head=True, head_width=6, head_length=2)
        if br_dx1 + br_dy1 != 0:
            ax.arrow(box[2], box[3], br_dx1, br_dy1, width=2, color=colour,
                     length_includes_head=True, head_width=6, head_length=2)
        if br_dx2 + br_dy2 != 0:
            ax.arrow(box[2], box[3], br_dx2, br_dy2, width=2, color=colour,
                     length_includes_head=True, head_width=6, head_length=2)

    elif mode == 'ellipse':

        draw_ellipse_corner(ax, (box[0], box[1]), width1=magnitude_tl2, height1=magnitude_tl1,
                            dx=tl_dx1, dy=tl_dy1, colour=colour)
        draw_ellipse_corner(ax, (box[2], box[3]), width1=magnitude_br2, height1=magnitude_br1,
                            dx=br_dx1, dy=br_dy1, colour=colour)


def draw_ellipse_corner(ax, centre, width1, height1, dx, dy, colour):
    # Calculate the angle to tilt the ellipses (largest eigenvector set to height)
    if dx == 0:
        tl_angle = 0
    elif dy == 0:
        tl_angle = 90
    else:
        tl_angle = 90 + np.arctan(dy / dx) * (180 / np.pi)

    ax.add_patch(Ellipse(centre, width=width1, height=height1, angle=tl_angle,
                         linewidth=2, edgecolor=colour, facecolor=None, fill=False))
    ax.add_patch(Ellipse(centre, width=2*width1, height=2*height1, angle=tl_angle,
                         linewidth=2, edgecolor=colour, facecolor=None, fill=False))
    ax.add_patch(Ellipse(centre, width=3*width1, height=3*height1, angle=tl_angle,
                         linewidth=2, edgecolor=colour, facecolor=None, fill=False))
