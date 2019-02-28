import numpy as np


def generate_bounding_box_from_mask(mask):
    flat_x = np.any(mask, axis=0)
    flat_y = np.any(mask, axis=1)
    if not np.any(flat_x) and not np.any(flat_y):
        raise ValueError("No positive pixels found, cannot compute bounding box")
    xmin = np.argmax(flat_x)
    ymin = np.argmax(flat_y)
    xmax = len(flat_x) - 1 - np.argmax(flat_x[::-1])
    ymax = len(flat_y) - 1 - np.argmax(flat_y[::-1])
    return [xmin, ymin, xmax, ymax]

def draw_cov(box, covs, ax, colour='k'):
    # TODO neaten this

    # Calculate the eigenvalues and eigenvectors for both covariances
    tl_vals, tl_vecs = np.linalg.eig(covs[0])
    br_vals, br_vecs = np.linalg.eig(covs[1])

    # flip the vectors along the y axis (for plotting as y is inverted)
    tl_vecs[:, 1] *= -1
    br_vecs[:, 1] *= -1

    # Determine which eigenvalue/vector is largest and which covariance that corresponds to
    tl_val_max_idx = np.argmax(tl_vals)
    tr_val_max_idx = np.argmax(br_vals)
    argmax_cov1 = np.argmax(np.diag(covs[0]))
    argmax_cov2 = np.argmax(np.diag(covs[1]))

    # Calculate the endpoint for arrows pointing in the direction of the eigenvectors
    # with length equal to std deviation along that vector

    # largest eigenvector
    magnitude_tl = np.sqrt(covs[0][argmax_cov1][argmax_cov1]) * 2
    magnitude_tr = np.sqrt(covs[1][argmax_cov2][argmax_cov2]) * 2
    tl_end1 = tl_vecs[tl_val_max_idx] * magnitude_tl + [box[0], box[1]]
    tr_end1 = br_vecs[tr_val_max_idx] * magnitude_tr + [box[2], box[3]]

    # smallest eigenvector
    tl_end2 = tl_vecs[invert_idx(tl_val_max_idx)] * np.sqrt(covs[0][invert_idx(argmax_cov1)]
                                                            [invert_idx(argmax_cov1)]
                                                            ) + [box[0], box[1]]
    tr_end2 = br_vecs[invert_idx(tr_val_max_idx)] * np.sqrt(covs[1][invert_idx(argmax_cov2)]
                                                            [invert_idx(argmax_cov2)]
                                                            ) + [box[2], box[3]]

    # Draw the arrows
    dx = tl_end1[0] - box[0]
    dy = tl_end1[1] - box[1]
    if not invalid_arrow(dx, dy):
        ax.arrow(box[0], box[1], dx, dy, width=2, color=colour, length_includes_head=True, head_width=6, head_length=2)
    dx = tl_end2[0] - box[0]
    dy = tl_end2[1] - box[1]
    if not invalid_arrow(dx, dy):
        ax.arrow(box[0], box[1], dx, dy, width=2, color=colour, length_includes_head=True, head_width=6, head_length=2)
    dx = tr_end1[0] - box[2]
    dy = tr_end1[1] - box[3]
    if not invalid_arrow(dx, dy):
        ax.arrow(box[2], box[3], dx, dy, width=2, color=colour, length_includes_head=True, head_width=6, head_length=2)
    dx = tr_end2[0] - box[2]
    dy = tr_end2[1] - box[3]
    if not invalid_arrow(dx, dy):
        ax.arrow(box[2], box[3], dx, dy, width=2, color=colour, length_includes_head=True, head_width=6, head_length=2)

def invert_idx(idx):
    return np.abs(1 - idx)


def invalid_arrow(dx, dy):
    if dx + dy == 0:
        return True
    return False